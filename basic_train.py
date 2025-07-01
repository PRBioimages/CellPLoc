from configs import Config
import torch
from utils.Evaluator import *
from utils.mix_methods import cutmix
from utils.softmatch_weight import SoftMatchWeighting
try:
    from apex import amp
except:
    pass
from losses.ASL import AsymmetricLoss


def cellweight(epoch, T1=5, T2=10, alpha=1):
    if epoch < T1:
        return 0
    elif epoch < T2:
        return 1.0 * (epoch-T1) * alpha / (T2 - T1)
    else:
        return alpha


def Pseudo_train_baseline(cfg: Config, model, train_dl, loss_func, optimizer, save_path, scheduler, writer, tune=None):
    print(f'[ ! ] pos weight: {cfg.loss.pos_weight}')
    print(f'[ ! ] cell weight: {cfg.loss.cellweight}')
    print(f'[ ! ] img weight: {cfg.loss.imgweight}')
    pos_weight = torch.ones(19).cuda() * cfg.loss.pos_weight
    print('[ √ ] Pseudo training')
    loss_func_cell = AsymmetricLoss(gamma_pos=0, clip_pos=0.01)
    loss_func_cellP = AsymmetricLoss(gamma_pos=0)
    if cfg.transform.size == 512:
        img_size = (600, 800)
    else:
        img_size = (cfg.transform.size, cfg.transform.size)
    try:
        optimizer.zero_grad()
        for epoch in range(cfg.train.start_epoch, cfg.train.num_epochs):
            # first we update batch sampler if exist
            if cfg.experiment.batch_sampler:
                train_dl.batch_sampler.update_miu(
                    cfg.experiment.initial_miu - epoch / cfg.experiment.miu_factor
                )
                print('[ W ] set miu to {}'.format(cfg.experiment.initial_miu - epoch / cfg.experiment.miu_factor))
            if scheduler and cfg.scheduler.name in ['StepLR']:
                scheduler.step(epoch)
            model.train()
            if not tune:
                tq = tqdm.tqdm(train_dl)
            else:
                tq = train_dl
            basic_lr = optimizer.param_groups[0]['lr']
            losses = []
            imageloss = []
            Calibrationloss = []
            pseudoloss = []
            # native amp
            if cfg.basic.amp == 'Native':
                scaler = torch.cuda.amp.GradScaler()
            for i, (ipt, mask, lbl, label) in enumerate(tq):
                ipt = ipt.view(-1, ipt.shape[-3], ipt.shape[-2], ipt.shape[-1])
                lbl = lbl.view(-1, lbl.shape[-1])
                image_label = label.cuda()
                # warm up lr initial
                if cfg.scheduler.warm_up and epoch == 0:
                    # warm up
                    length = len(train_dl)
                    initial_lr = basic_lr / length
                    optimizer.param_groups[0]['lr'] = initial_lr * (i + 1)
                ipt, lbl = ipt.cuda(), lbl.cuda()
                if cfg.basic.amp == 'Native':
                    with torch.cuda.amp.autocast():
                        cell, image, pseudo_cell = model(ipt, cfg.experiment.count)
                        loss_cell = loss_func_cell(cell, lbl)

                        pseudo_cell_label = torch.sigmoid(pseudo_cell.detach())
                        pseudo_cell_label = pseudo_cell_label * lbl
                        loss_cell_pseudo = loss_func_cellP(cell.float(), pseudo_cell_label.float())

                        loss_exp = loss_func(image, image_label)

                        alpha = cellweight(epoch, alpha=cfg.experiment.alpha)

                        if not len(loss_cell.shape) == 0:
                            loss_cell = loss_cell.mean()
                        if not len(loss_exp.shape) == 0:
                            loss_exp = loss_exp.mean()
                        if not len(loss_cell_pseudo.shape) == 0:
                            loss_cell_pseudo = loss_cell_pseudo.mean()

                        loss = cfg.loss.cellweight * ((1-alpha) * loss_cell + alpha * loss_cell_pseudo) + \
                               cfg.loss.imgweight * loss_exp

                        losses.append(loss.item())
                        imageloss.append(loss_exp.item())
                        Calibrationloss.append(loss_cell.item())
                        pseudoloss.append(loss_cell_pseudo.item())

                if cfg.basic.amp == 'Native':
                    scaler.scale(loss).backward()
                elif not cfg.basic.amp == 'None':
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if i % cfg.optimizer.step == 0:
                    if cfg.basic.amp == 'Native':
                        if cfg.train.clip:
                            scaler.unscale_(optimizer)
                            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    else:
                        if cfg.train.clip:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip)
                        optimizer.step()
                        optimizer.zero_grad()
                if cfg.scheduler.name in ['CyclicLR', 'OneCycleLR', 'CosineAnnealingLR']:
                    if epoch == 0 and cfg.scheduler.warm_up:
                        pass
                    else:
                    # TODO maybe, a bug
                        scheduler.step()
                if not tune:
                    info = {'Iloss':np.array(imageloss).mean(), 'Closs':np.array(Calibrationloss).mean(),
                            'ploss': np.mean(pseudoloss)}
                    tq.set_postfix(info)
            if len(cfg.basic.GPU) > 1:
                torch.save(model.module.state_dict(), save_path / 'checkpoints/f{}_epoch-{}.pth'.format(
                    cfg.experiment.run_fold, epoch))
            else:
                torch.save(model.state_dict(), save_path / 'checkpoints/f{}_epoch-{}.pth'.format(
                    cfg.experiment.run_fold, epoch))

            print(('[ √ ] epochs: {}, train loss: {:.4f}, p loss: {:.4f}').format(epoch, np.array(losses).mean(),
                                                                                  np.array(pseudoloss).mean()))

            writer.add_scalar('train_f{}/loss'.format(cfg.experiment.run_fold), np.mean(losses), epoch)
            writer.add_scalar('train_f{}/cell_loss'.format(cfg.experiment.run_fold), np.mean(Calibrationloss), epoch)
            writer.add_scalar('train_f{}/img_loss'.format(cfg.experiment.run_fold), np.mean(imageloss), epoch)
            writer.add_scalar('train_f{}/p_loss'.format(cfg.experiment.run_fold), np.mean(pseudoloss), epoch)
            writer.add_scalar('train_f{}/lr'.format(cfg.experiment.run_fold), optimizer.param_groups[0]['lr'], epoch)

            with open(save_path / 'train.log', 'a') as fp:
                fp.write('{}\t{:.8f}\t{:.6f}\t{:.8f}\t{:.4f}\t{:.4f}\n'.format(
                    epoch, optimizer.param_groups[0]['lr'], np.array(losses).mean(), np.array(Calibrationloss).mean(),
                    np.array(imageloss).mean(), np.array(pseudoloss).mean()))

    except KeyboardInterrupt:
        print('[ X ] Ctrl + c, QUIT')
        if len(cfg.basic.GPU) > 1:
            torch.save(model.module.state_dict(), save_path / 'checkpoints/quit{}_epoch-{}.pth'.format(
                cfg.experiment.run_fold, epoch))
        else:
            torch.save(model.state_dict(), save_path / 'checkpoints/quit{}_epoch-{}.pth'.format(
                cfg.experiment.run_fold, epoch))


def Pseudo_train_HPAv21(cfg: Config, model, train_dl, loss_func, optimizer, save_path, scheduler, writer, tune=None):
    print(f'[ ! ] pos weight: {cfg.loss.pos_weight}')  # 1
    print(f'[ ! ] cell weight: {cfg.loss.cellweight}')  # 0.1
    print(f'[ ! ] img weight: {cfg.loss.imgweight}')  # 1
    pos_weight = torch.ones(19).cuda() * cfg.loss.pos_weight
    print('[ √ ] Pseudo training')

    softmatch_weight_pseudo_cell = SoftMatchWeighting(num_classes=cfg.softmatch.num_classes,
                                                      n_sigma=cfg.softmatch.n_sigma,
                                                      momentum=cfg.softmatch.ema_p, per_class=cfg.softmatch.per_class)
    softmatch_weight_biblabel = SoftMatchWeighting(num_classes=cfg.softmatch.num_classes,
                                                   n_sigma=cfg.softmatch.n_sigma,
                                                   momentum=cfg.softmatch.ema_p, per_class=cfg.softmatch.per_class)

    loss_func_cell = AsymmetricLoss(gamma_pos=0, clip_pos=0.01)
    loss_func_cellP = AsymmetricLoss(gamma_pos=0)
    if cfg.transform.size == 512:
        img_size = (600, 800)
    else:
        img_size = (cfg.transform.size, cfg.transform.size)
    try:
        optimizer.zero_grad()
        for epoch in range(cfg.train.start_epoch, cfg.train.num_epochs):
            print(f'[ ! ] Start Epoch{epoch}')
            # first we update batch sampler if exist
            if cfg.experiment.batch_sampler:
                train_dl.batch_sampler.update_miu(
                    cfg.experiment.initial_miu - epoch / cfg.experiment.miu_factor
                )
                print('[ W ] set miu to {}'.format(cfg.experiment.initial_miu - epoch / cfg.experiment.miu_factor))
            if scheduler and cfg.scheduler.name in ['StepLR']:
                scheduler.step(epoch)
            model.train()
            if not tune:
                tq = tqdm.tqdm(train_dl, ncols=100)
            else:
                tq = train_dl
            basic_lr = optimizer.param_groups[0]['lr']
            losses = []
            losses_cla = []
            imageloss = []
            Calibrationloss = []
            pseudoloss = []
            losses_con = []
            imageloss_con = []
            Calibrationloss_con = []
            pseudoloss_con = []
            # native amp
            if cfg.basic.amp == 'Native':
                scaler = torch.cuda.amp.GradScaler()
            for i, (ipt, ipt_s, mask, lbl, label, bib_lbl) in enumerate(tq):
                ipt = ipt.view(-1, ipt.shape[-3], ipt.shape[-2], ipt.shape[-1])
                ipt_s = ipt_s.view(-1, ipt_s.shape[-3], ipt_s.shape[-2], ipt_s.shape[-1])
                lbl = lbl.view(-1, lbl.shape[-1])
                bib_lbl = bib_lbl.view(-1, bib_lbl.shape[-1])
                image_label = label.cuda()
                # warm up lr initial
                if cfg.scheduler.warm_up and epoch == 0:
                    # warm up
                    length = len(train_dl)
                    initial_lr = basic_lr / length
                    optimizer.param_groups[0]['lr'] = initial_lr * (i + 1)
                ipt, lbl = ipt.cuda(), lbl.cuda()
                num_lb = ipt.size(0)  # 获取弱增强数据的批处理大小
                ipt_s = ipt_s.cuda()
                bib_lbl = bib_lbl.cuda()
                if cfg.basic.amp == 'Native':
                    with torch.cuda.amp.autocast():
                        inputs = torch.cat((ipt, ipt_s))

                        cell, image, pseudo_cell = model(inputs, cfg.experiment.count)

                        # 拆分输出为弱增强部分与强增强部分
                        cell_w = cell[:num_lb]
                        cell_s = cell[num_lb:]

                        with torch.no_grad():
                            size_image = image.shape[0]
                            size_image = size_image * 0.5

                        image_w = image[:int(size_image)]
                        image_s = image[int(size_image):]

                        pseudo_cell_w = pseudo_cell[:num_lb]
                        pseudo_cell_s = pseudo_cell[num_lb:]

                        cell_w_sigmoid = torch.sigmoid(cell_w.detach())
                        pseudo_cell_w_sigmoid = torch.sigmoid(pseudo_cell_w.detach())
                        image_w_sigmoid = torch.sigmoid(image_w.detach())

                        cell_w_sigmoid_lbl = cell_w_sigmoid * lbl
                        image_w_sigmoid_lbl = image_w_sigmoid * image_label
                        pseudo_cell_w_sigmoid_lbl = pseudo_cell_w_sigmoid * lbl

                        pseudo_weight_pseudo_cell = softmatch_weight_pseudo_cell.masking(
                            logits_x_ulb=pseudo_cell_w_sigmoid_lbl.float(), labels=lbl, sigmoid_x_ulb=False)

                        pseudo_weight_biblabel = softmatch_weight_biblabel.masking(
                            logits_x_ulb=bib_lbl.float(), labels=lbl, sigmoid_x_ulb=False)

                        # 添加硬阈值机制，将masking里不为1的值置零
                        if cfg.experiment.use_hard_mask:
                            # 创建一个与 pseudo_weight 形状相同的布尔张量，其中值不为1的位置为True
                            pseudo_weight_pseudo_cell_mask = (pseudo_weight_pseudo_cell != 1)
                            pseudo_weight_biblabel_mask = (pseudo_weight_biblabel != 1)
                            # 使用布尔张量来置零
                            pseudo_weight_pseudo_cell[pseudo_weight_pseudo_cell_mask] = 0
                            pseudo_weight_biblabel[pseudo_weight_biblabel_mask] = 0

                        loss_cell_bib = loss_func_cell(cell_w.float(), bib_lbl.float())
                        loss_cell_bib = loss_cell_bib.float() * pseudo_weight_biblabel.float()

                        loss_cell_img = loss_func_cell(cell_w.float(), lbl.float())
                        loss_cell = loss_cell_bib.float() * 0.8 + loss_cell_img.float() * 0.2
                        loss_cell_pseudo = loss_func_cellP(cell_w.float(), pseudo_cell_w_sigmoid_lbl.float())
                        loss_cell_pseudo = loss_cell_pseudo.float() * pseudo_weight_pseudo_cell.float()

                        loss_exp = loss_func(image_w.float(), image_label.float())

                        if cfg.experiment.use_hard_label:
                            consistency_loss_cell = torch.nn.BCEWithLogitsLoss(reduction="none")(
                                cell_s.float(),
                                cell_w_sigmoid_lbl.float())
                            consistency_loss_exp = torch.nn.BCEWithLogitsLoss(reduction="none")(
                                image_s.float(),
                                image_w_sigmoid_lbl.float())
                            consistency_loss_cell_pseudo = torch.nn.BCEWithLogitsLoss(reduction="none")(
                                pseudo_cell_s.float(),
                                pseudo_cell_w_sigmoid_lbl.float())
                        else:
                            consistency_loss_cell = torch.nn.BCEWithLogitsLoss(reduction="none")(
                                cell_s.float(),
                                cell_w_sigmoid.float())
                            consistency_loss_exp = torch.nn.BCEWithLogitsLoss(reduction="none")(
                                image_s.float(),
                                image_w_sigmoid.float())
                            consistency_loss_cell_pseudo = torch.nn.BCEWithLogitsLoss(reduction="none")(
                                pseudo_cell_s.float(),
                                pseudo_cell_w_sigmoid.float())

                        alpha = cellweight(epoch, alpha=cfg.experiment.alpha)


                        if not len(loss_cell.shape) == 0:
                            loss_cell = loss_cell.mean()
                        if not len(loss_exp.shape) == 0:
                            loss_exp = loss_exp.mean()
                        if not len(loss_cell_pseudo.shape) == 0:
                            loss_cell_pseudo = loss_cell_pseudo.mean()

                        if not len(consistency_loss_cell.shape) == 0:
                            consistency_loss_cell = consistency_loss_cell.mean()
                        if not len(consistency_loss_exp.shape) == 0:
                            consistency_loss_exp = consistency_loss_exp.mean()
                        if not len(consistency_loss_cell_pseudo.shape) == 0:
                            consistency_loss_cell_pseudo = consistency_loss_cell_pseudo.mean()

                        classification_loss = cfg.loss.cellweight * ((1-alpha) * loss_cell.float() + alpha * loss_cell_pseudo.float()) + \
                               cfg.loss.imgweight * loss_exp.float()  # cellweight=0.1, imgweight=1

                        consistency_loss = (consistency_loss_exp + consistency_loss_cell + consistency_loss_cell_pseudo) * 0.05

                        loss = classification_loss.float() + consistency_loss.float()
                        if cfg.optimizer.step != 1:
                            loss = loss / cfg.optimizer.step

                        losses.append(loss.item())

                        losses_cla.append(classification_loss.item())
                        imageloss.append(loss_exp.item())
                        Calibrationloss.append(loss_cell.item())
                        pseudoloss.append(loss_cell_pseudo.item())

                        losses_con.append(consistency_loss.item())
                        imageloss_con.append(consistency_loss_exp.item())
                        Calibrationloss_con.append(consistency_loss_cell.item())
                        pseudoloss_con.append(consistency_loss_cell_pseudo.item())

                if cfg.basic.amp == 'Native':
                    scaler.scale(loss).backward()
                elif not cfg.basic.amp == 'None':
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                if (i+1) % cfg.optimizer.step == 0:
                    if cfg.basic.amp == 'Native':
                        if cfg.train.clip:
                            scaler.unscale_(optimizer)
                            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    else:
                        if cfg.train.clip:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip)
                        optimizer.step()
                        optimizer.zero_grad()
                if cfg.scheduler.name in ['CyclicLR', 'OneCycleLR', 'CosineAnnealingLR']:
                    if epoch == 0 and cfg.scheduler.warm_up:
                        pass
                    else:
                    # TODO maybe, a bug
                        scheduler.step()
                if not tune:
                    info = {'cla_loss': np.array(losses_cla).mean(), 'cr_loss': np.array(losses_con).mean(),
                            'total_loss': np.array(losses).mean()}
                    tq.set_postfix(info)
            if len(cfg.basic.GPU) > 1:
                torch.save(model.module.state_dict(), save_path / 'checkpoints/f{}_epoch-{}.pth'.format(
                    cfg.experiment.run_fold, epoch))
            else:
                torch.save(model.state_dict(), save_path / 'checkpoints/f{}_epoch-{}.pth'.format(
                    cfg.experiment.run_fold, epoch))

            print(('[ √ ] epochs: {}, train loss: {:.4f}').format(epoch, np.array(losses).mean()))
            print(('[ √ ] classification_loss: {:.4f}, consistency_loss: {:.4f}').format(np.array(losses_cla).mean(),
                                                                                         np.array(losses_con).mean()))

            writer.add_scalar('train_f{}/loss'.format(cfg.experiment.run_fold), np.mean(losses), epoch)

            writer.add_scalar('train_f{}/classification_loss'.format(cfg.experiment.run_fold), np.mean(losses_cla), epoch)
            writer.add_scalar('train_f{}/cell_loss'.format(cfg.experiment.run_fold), np.mean(Calibrationloss), epoch)
            writer.add_scalar('train_f{}/img_loss'.format(cfg.experiment.run_fold), np.mean(imageloss), epoch)
            writer.add_scalar('train_f{}/p_loss'.format(cfg.experiment.run_fold), np.mean(pseudoloss), epoch)
            writer.add_scalar('train_f{}/lr'.format(cfg.experiment.run_fold), optimizer.param_groups[0]['lr'], epoch)

            writer.add_scalar('train_f{}/consistency_loss'.format(cfg.experiment.run_fold), np.mean(losses_con), epoch)
            writer.add_scalar('train_f{}/consistency_cell_loss'.format(cfg.experiment.run_fold), np.mean(Calibrationloss_con), epoch)
            writer.add_scalar('train_f{}/consistency_img_loss'.format(cfg.experiment.run_fold), np.mean(imageloss_con), epoch)
            writer.add_scalar('train_f{}/consistency_p_loss'.format(cfg.experiment.run_fold), np.mean(pseudoloss_con), epoch)

            with open(save_path / 'logs' / 'train.log', 'a') as fp:
                fp.write('{}\t{:.8f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.4f}\t{:.6f}\t{:.12f}\t{:.8f}\t{:.8f}\n'
                         .format(epoch, optimizer.param_groups[0]['lr'], np.array(losses).mean(),
                                 np.array(losses_cla).mean(), np.array(Calibrationloss).mean(),
                                 np.array(imageloss).mean(), np.array(pseudoloss).mean(),
                                 np.array(losses_con).mean(), np.array(Calibrationloss_con).mean(),
                                 np.array(imageloss_con).mean(), np.array(pseudoloss_con).mean()))
        # # 在训练结束后找到最佳的mAP值和对应的epoch
        # epoch_mAP_pairs = enumerate(mAP_values, start=0)
        # best_epoch, best_mAP = max(epoch_mAP_pairs, key=lambda x: x[1])
        #
        # print(f"模型的最佳epoch是: {best_epoch}, mAP值为: {best_mAP}")

    except KeyboardInterrupt:
        print('[ X ] Ctrl + c, QUIT')
        if len(cfg.basic.GPU) > 1:
            torch.save(model.module.state_dict(), save_path / 'checkpoints/quit{}_epoch-{}.pth'.format(
                cfg.experiment.run_fold, epoch))
        else:
            torch.save(model.state_dict(), save_path / 'checkpoints/quit{}_epoch-{}.pth'.format(
                cfg.experiment.run_fold, epoch))
        with open(save_path / 'logs' / 'train.log', 'a') as fp:
            fp.write('{}\t{:.8f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.4f}\t{:.6f}\t{:.12f}\t{:.8f}\t{:.8f}\n'
                     .format(epoch, optimizer.param_groups[0]['lr'], np.array(losses).mean(),
                             np.array(losses_cla).mean(), np.array(Calibrationloss).mean(),
                             np.array(imageloss).mean(), np.array(pseudoloss).mean(),
                             np.array(losses_con).mean(), np.array(Calibrationloss_con).mean(),
                             np.array(imageloss_con).mean(), np.array(pseudoloss_con).mean()))


def Pseudo_train_HPAv23(cfg: Config, model, train_dl, loss_func, optimizer, save_path, scheduler, writer, tune=None):
    print(f'[ ! ] pos weight: {cfg.loss.pos_weight}')  # 1
    print(f'[ ! ] cell weight: {cfg.loss.cellweight}')  # 0.1
    print(f'[ ! ] img weight: {cfg.loss.imgweight}')  # 1
    pos_weight = torch.ones(19).cuda() * cfg.loss.pos_weight
    print('[ √ ] Pseudo training, using HPAv23 datasets')

    softmatch_weight_pseudo_cell = SoftMatchWeighting(num_classes=cfg.softmatch.num_classes,
                                                      n_sigma=cfg.softmatch.n_sigma,
                                                      momentum=cfg.softmatch.ema_p, per_class=cfg.softmatch.per_class)

    loss_func_cell = AsymmetricLoss(gamma_pos=0, clip_pos=0.01)
    loss_func_cellP = AsymmetricLoss(gamma_pos=0)
    if cfg.transform.size == 512:
        img_size = (600, 800)
    else:
        img_size = (cfg.transform.size, cfg.transform.size)
    try:
        optimizer.zero_grad()
        for epoch in range(cfg.train.start_epoch, cfg.train.num_epochs):
            print(f'[ ! ] Start Epoch{epoch}')
            # first we update batch sampler if exist
            if cfg.experiment.batch_sampler:
                train_dl.batch_sampler.update_miu(
                    cfg.experiment.initial_miu - epoch / cfg.experiment.miu_factor
                )
                print('[ W ] set miu to {}'.format(cfg.experiment.initial_miu - epoch / cfg.experiment.miu_factor))
            if scheduler and cfg.scheduler.name in ['StepLR']:
                scheduler.step(epoch)
            model.train()
            if not tune:
                tq = tqdm.tqdm(train_dl, ncols=100)
            else:
                tq = train_dl
            basic_lr = optimizer.param_groups[0]['lr']
            losses = []
            losses_cla = []
            imageloss = []
            Calibrationloss = []
            pseudoloss = []
            losses_con = []
            imageloss_con = []
            Calibrationloss_con = []
            pseudoloss_con = []
            # native amp
            if cfg.basic.amp == 'Native':
                scaler = torch.cuda.amp.GradScaler()
            for i, (ipt, ipt_s, mask, lbl, label) in enumerate(tq):
                ipt = ipt.view(-1, ipt.shape[-3], ipt.shape[-2], ipt.shape[-1])
                ipt_s = ipt_s.view(-1, ipt_s.shape[-3], ipt_s.shape[-2], ipt_s.shape[-1])
                lbl = lbl.view(-1, lbl.shape[-1])
                image_label = label.cuda()
                # warm up lr initial
                if cfg.scheduler.warm_up and epoch == 0:
                    # warm up
                    length = len(train_dl)
                    initial_lr = basic_lr / length
                    optimizer.param_groups[0]['lr'] = initial_lr * (i + 1)
                ipt, lbl = ipt.cuda(), lbl.cuda()
                num_lb = ipt.size(0)  # 获取弱增强数据的批处理大小
                ipt_s = ipt_s.cuda()
                if cfg.basic.amp == 'Native':
                    with torch.cuda.amp.autocast():
                        inputs = torch.cat((ipt, ipt_s))

                        cell, image, pseudo_cell = model(inputs, cfg.experiment.count)

                        # 拆分输出为弱增强部分与强增强部分
                        cell_w = cell[:num_lb]
                        cell_s = cell[num_lb:]

                        with torch.no_grad():
                            size_image = image.shape[0]
                            size_image = size_image * 0.5

                        image_w = image[:int(size_image)]
                        image_s = image[int(size_image):]

                        pseudo_cell_w = pseudo_cell[:num_lb]
                        pseudo_cell_s = pseudo_cell[num_lb:]

                        cell_w_sigmoid = torch.sigmoid(cell_w.detach())
                        pseudo_cell_w_sigmoid = torch.sigmoid(pseudo_cell_w.detach())
                        image_w_sigmoid = torch.sigmoid(image_w.detach())

                        cell_w_sigmoid_lbl = cell_w_sigmoid * lbl
                        image_w_sigmoid_lbl = image_w_sigmoid * image_label
                        pseudo_cell_w_sigmoid_lbl = pseudo_cell_w_sigmoid * lbl

                        pseudo_weight_pseudo_cell = softmatch_weight_pseudo_cell.masking(
                            logits_x_ulb=pseudo_cell_w_sigmoid_lbl.float(), labels=lbl, sigmoid_x_ulb=False)

                        # 添加硬阈值机制，将masking里不为1的值置零
                        if cfg.experiment.use_hard_mask:
                            # 创建一个与 pseudo_weight 形状相同的布尔张量，其中值不为1的位置为True
                            pseudo_weight_pseudo_cell_mask = (pseudo_weight_pseudo_cell != 1)
                            # 使用布尔张量来置零
                            pseudo_weight_pseudo_cell[pseudo_weight_pseudo_cell_mask] = 0

                        loss_cell_img = loss_func_cell(cell_w.float(), lbl.float())
                        loss_cell = loss_cell_img.float() * 1.0

                        loss_cell_pseudo = loss_func_cellP(cell_w.float(), pseudo_cell_w_sigmoid_lbl.float())
                        loss_cell_pseudo = loss_cell_pseudo.float() * pseudo_weight_pseudo_cell.float()

                        loss_exp = loss_func(image_w.float(), image_label.float())

                        if cfg.experiment.use_hard_label:
                            consistency_loss_cell = torch.nn.BCEWithLogitsLoss(reduction="none")(
                                cell_s.float(),
                                cell_w_sigmoid_lbl.float())
                            consistency_loss_exp = torch.nn.BCEWithLogitsLoss(reduction="none")(
                                image_s.float(),
                                image_w_sigmoid_lbl.float())
                            consistency_loss_cell_pseudo = torch.nn.BCEWithLogitsLoss(reduction="none")(
                                pseudo_cell_s.float(),
                                pseudo_cell_w_sigmoid_lbl.float())
                        else:
                            consistency_loss_cell = torch.nn.BCEWithLogitsLoss(reduction="none")(
                                cell_s.float(),
                                cell_w_sigmoid.float())
                            consistency_loss_exp = torch.nn.BCEWithLogitsLoss(reduction="none")(
                                image_s.float(),
                                image_w_sigmoid.float())
                            consistency_loss_cell_pseudo = torch.nn.BCEWithLogitsLoss(reduction="none")(
                                pseudo_cell_s.float(),
                                pseudo_cell_w_sigmoid.float())

                        alpha = cellweight(epoch, alpha=cfg.experiment.alpha)

                        if not len(loss_cell.shape) == 0:
                            loss_cell = loss_cell.mean()
                        if not len(loss_exp.shape) == 0:
                            loss_exp = loss_exp.mean()
                        if not len(loss_cell_pseudo.shape) == 0:
                            loss_cell_pseudo = loss_cell_pseudo.mean()

                        if not len(consistency_loss_cell.shape) == 0:
                            consistency_loss_cell = consistency_loss_cell.mean()
                        if not len(consistency_loss_exp.shape) == 0:
                            consistency_loss_exp = consistency_loss_exp.mean()
                        if not len(consistency_loss_cell_pseudo.shape) == 0:
                            consistency_loss_cell_pseudo = consistency_loss_cell_pseudo.mean()

                        classification_loss = \
                            cfg.loss.cellweight * ((1-alpha) * loss_cell.float() + alpha * loss_cell_pseudo.float()) \
                            + cfg.loss.imgweight * loss_exp.float()  # cellweight=0.1, imgweight=1

                        consistency_loss = \
                            (consistency_loss_exp + consistency_loss_cell + consistency_loss_cell_pseudo) * 0.05

                        loss = classification_loss.float() + consistency_loss.float()
                        if cfg.optimizer.step != 1:
                            loss = loss / cfg.optimizer.step

                        losses.append(loss.item())

                        losses_cla.append(classification_loss.item())
                        imageloss.append(loss_exp.item())
                        Calibrationloss.append(loss_cell.item())
                        pseudoloss.append(loss_cell_pseudo.item())

                        losses_con.append(consistency_loss.item())
                        imageloss_con.append(consistency_loss_exp.item())
                        Calibrationloss_con.append(consistency_loss_cell.item())
                        pseudoloss_con.append(consistency_loss_cell_pseudo.item())

                if cfg.basic.amp == 'Native':
                    scaler.scale(loss).backward()
                elif not cfg.basic.amp == 'None':
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                if (i+1) % cfg.optimizer.step == 0:
                    if cfg.basic.amp == 'Native':
                        if cfg.train.clip:
                            scaler.unscale_(optimizer)
                            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    else:
                        if cfg.train.clip:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip)
                        optimizer.step()
                        optimizer.zero_grad()
                if cfg.scheduler.name in ['CyclicLR', 'OneCycleLR', 'CosineAnnealingLR']:
                    if epoch == 0 and cfg.scheduler.warm_up:
                        pass
                    else:
                    # TODO maybe, a bug
                        scheduler.step()
                if not tune:
                    info = {'cla_loss': np.array(losses_cla).mean(), 'cr_loss': np.array(losses_con).mean(),
                            'total_loss': np.array(losses).mean()}
                    tq.set_postfix(info)
            if len(cfg.basic.GPU) > 1:
                torch.save(model.module.state_dict(), save_path / 'checkpoints/f{}_epoch-{}.pth'.format(
                    cfg.experiment.run_fold, epoch))
            else:
                torch.save(model.state_dict(), save_path / 'checkpoints/f{}_epoch-{}.pth'.format(
                    cfg.experiment.run_fold, epoch))

            print('[ √ ] epochs: {}, train loss: {:.4f}'.format(epoch, np.array(losses).mean()))
            print('[ √ ] classification_loss: {:.4f}, consistency_loss: {:.4f}'.format(np.array(losses_cla).mean(),
                                                                                       np.array(losses_con).mean()))

            writer.add_scalar('train_f{}/loss'.format(cfg.experiment.run_fold), np.mean(losses), epoch)

            writer.add_scalar('train_f{}/classification_loss'.format(cfg.experiment.run_fold), np.mean(losses_cla), epoch)
            writer.add_scalar('train_f{}/cell_loss'.format(cfg.experiment.run_fold), np.mean(Calibrationloss), epoch)
            writer.add_scalar('train_f{}/img_loss'.format(cfg.experiment.run_fold), np.mean(imageloss), epoch)
            writer.add_scalar('train_f{}/p_loss'.format(cfg.experiment.run_fold), np.mean(pseudoloss), epoch)
            writer.add_scalar('train_f{}/lr'.format(cfg.experiment.run_fold), optimizer.param_groups[0]['lr'], epoch)

            writer.add_scalar('train_f{}/consistency_loss'.format(cfg.experiment.run_fold), np.mean(losses_con), epoch)
            writer.add_scalar('train_f{}/consistency_cell_loss'.format(cfg.experiment.run_fold), np.mean(Calibrationloss_con), epoch)
            writer.add_scalar('train_f{}/consistency_img_loss'.format(cfg.experiment.run_fold), np.mean(imageloss_con), epoch)
            writer.add_scalar('train_f{}/consistency_p_loss'.format(cfg.experiment.run_fold), np.mean(pseudoloss_con), epoch)

            with open(save_path / 'logs' / 'train.log', 'a') as fp:
                fp.write('{}\t{:.8f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.4f}\t{:.6f}\t{:.12f}\t{:.8f}\t{:.8f}\n'
                         .format(epoch, optimizer.param_groups[0]['lr'], np.array(losses).mean(),
                                 np.array(losses_cla).mean(), np.array(Calibrationloss).mean(),
                                 np.array(imageloss).mean(), np.array(pseudoloss).mean(),
                                 np.array(losses_con).mean(), np.array(Calibrationloss_con).mean(),
                                 np.array(imageloss_con).mean(), np.array(pseudoloss_con).mean()))

    except KeyboardInterrupt:
        print('[ X ] Ctrl + c, QUIT')
        if len(cfg.basic.GPU) > 1:
            torch.save(model.module.state_dict(), save_path / 'checkpoints/quit{}_epoch-{}.pth'.format(
                cfg.experiment.run_fold, epoch))
        else:
            torch.save(model.state_dict(), save_path / 'checkpoints/quit{}_epoch-{}.pth'.format(
                cfg.experiment.run_fold, epoch))
        with open(save_path / 'logs' / 'train.log', 'a') as fp:
            fp.write('{}\t{:.8f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.4f}\t{:.6f}\t{:.12f}\t{:.8f}\t{:.8f}\n'
                     .format(epoch, optimizer.param_groups[0]['lr'], np.array(losses).mean(),
                             np.array(losses_cla).mean(), np.array(Calibrationloss).mean(),
                             np.array(imageloss).mean(), np.array(pseudoloss).mean(),
                             np.array(losses_con).mean(), np.array(Calibrationloss_con).mean(),
                             np.array(imageloss_con).mean(), np.array(pseudoloss_con).mean()))

