from utils import *
import numpy as np
import time
from config import args



def validation(x_valid, y_valid, val_batch_size, num_classes, sess, model, epoch, start_time, w_plus):
    loss_batch_all = np.array([])
    acc_batch_all = y_pred_all = logits_all = np.zeros((0, num_classes))
    model.is_train = False
    x_valid, y_valid = randomize(x_valid, y_valid)
    step_count = int(len(x_valid) / val_batch_size)

    for step in range(step_count):
        start = step * val_batch_size
        end = (step + 1) * val_batch_size
        x_batch, y_batch = get_next_batch(x_valid, y_valid, start, end)

        feed_dict_val = {model.x: x_batch, model.y: y_batch, model.w_plus: w_plus}
        acc_valid, loss_valid, y_pred, logits = sess.run(
            [model.accuracy, model.loss, model.prediction, model.get_logits],
            feed_dict=feed_dict_val)

        acc_batch_all = np.concatenate((acc_batch_all, acc_valid.reshape([1, num_classes])))
        y_pred_all = np.concatenate((y_pred_all, y_pred.reshape([val_batch_size, num_classes])))
        logits_all = np.concatenate((logits_all, logits.reshape([val_batch_size, num_classes])))
        loss_batch_all = np.append(loss_batch_all, loss_valid)

    mean_acc = np.mean(acc_batch_all, axis=0)
    mean_loss = np.mean(loss_batch_all)
    num_examples = np.sum(y_valid, axis=0)
    num_preds = np.sum(y_pred_all, axis=0)
    epoch_time = time.time() - start_time
    print('******************************************************************************'
          '********************************************************')
    print('--------------------------------------------------------Validation, Epoch: {}'
          ' -----------------------------------------------------------'.format(epoch + 1))
    print("Atlc\tCrdmg\tEffus\tInflt\tMass\tNodle\tPnum\tPntrx\tConsd"
          "\tEdma\tEmpys\tFbrss\tTkng\tHrna\t|Avg.\t|Loss\t|Run Time")
    for accu in mean_acc:
        print '{:.01%}\t'.format(accu),
    print '|{0:.01%}\t|{1:0.02}\t|{2}'.format(np.mean(mean_acc), mean_loss, epoch_time)

    for exm in num_examples:
        print '{:}\t'.format(exm),
    print("Count of pathalogies")
    for pred in num_preds:
        print '{:}\t'.format(pred),
    print("Count of recognized pathalogies")

    P = R = np.zeros((1, args.n_cls))
    for cond in range(args.n_cls):
        y_true = y_valid[:, cond]
        y_pred = y_pred_all[:, cond]
        y_true = y_true[:len(y_pred)]
        P[0, cond], R[0, cond]= precision_recall(y_true, y_pred)
    P = np.reshape(P, args.n_cls)
    R = np.reshape(R, args.n_cls)

    for p in P:
        print '{:0.03}\t'.format(p),
    print("Precision")
    for r in R:
        print '{:0.03}\t'.format(r),
    print("Recall")

    plot_precision_recall_curve(y_valid[:logits_all.shape[0], :], logits_all, epoch)
    write_acc_loss_csv(mean_acc, mean_loss, epoch)
    write_precision_recall_csv(P, R, epoch)

    return mean_acc, mean_loss