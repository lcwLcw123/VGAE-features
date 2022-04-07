import time
import copy
import pickle
import warnings
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def mask_test_features(features_orig, val_frac, test_frac, no_mask):
    features =features_orig

    features_triu = sp.triu(features)
    print('features_triu')
    print(features_triu)
    features_tuple = sparse_to_tuple(features_triu)[0]
    print('features_tuple')
    features_all = sparse_to_tuple(features)[0]
    print('features_all')
    print(features_all)
    num_test = int(np.floor(features_tuple.shape[0] * test_frac))
    num_val = int(np.floor(features_tuple.shape[0] * val_frac))
    print(np.floor(features_tuple.shape[0]))

    all_edge_idx = list(range(features_tuple.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = features_tuple[test_edge_idx]
    val_edges = features_tuple[val_edge_idx]
    if no_mask:
        train_features = features_tuple
        print('no_mask')
    else:
        train_features = np.delete(features_tuple, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
        print('finish mask features')

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, features.shape[0])
        idx_j = np.random.randint(0, features.shape[1])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], features_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, features.shape[0])
        idx_j = np.random.randint(0, features.shape[1])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_features):
            continue
        if ismember([idx_j, idx_i], train_features):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    # assert ~ismember(test_edges_false, edges_all)
    # assert ~ismember(val_edges_false, edges_all)
    # assert ~ismember(val_edges, test_edges)
    # if not no_mask:
    #     assert ~ismember(val_edges, train_edges)
    #     assert ~ismember(test_edges, train_edges)

    # Re-build adj matrix
    features_train = sp.csr_matrix((np.ones(train_features.shape[0]), (train_features[:, 0], train_features[:, 1])), shape=features.shape)
    # NOTE: these edge lists only contain single direction of edge!
    val_features = val_edges
    val_features_false = np.asarray(val_edges_false)
    test_features = test_edges
    test_features_false = np.asarray(test_edges_false)

    return features_train,val_features,val_features_false,test_features,test_features_false

    # assert ~ismember(test_edges_false, edges_all)
    # assert ~ismember(val_edges_false, edges_all)
    # assert ~ismember(val_edges, test_edges)
    # if not no_mask:
    #     assert ~ismember(val_edges, train_edges)
    #     assert ~ismember(test_edges, train_edges)

    # Re-build adj matrix
    # adj_train = sp.csr_ma
    # trix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    # self.adj_train = adj_train + adj_train.T
    # self.adj_label = adj_train + sp.eye(adj_train.shape[0])
    # # NOTE: these edge lists only contain single direction of edge!
    # self.val_edges = val_edges
    # self.val_edges_false = np.asarray(val_edges_false)
    # self.test_edges = test_edges
    # self.test_edges_false = np.asarray(test_edges_false)

def get_scores_features(val_features,val_features_false,Features_pred, features):
    # get logists and labels
    preds = Features_pred[val_features.T]

    preds_neg = Features_pred[val_features_false.T]
    logists = np.hstack([preds, preds_neg])
    labels = np.hstack([np.ones(preds.size(0)), np.zeros(preds_neg.size(0))])
    # logists = A_pred.view(-1)
    # labels = adj_label.to_dense().view(-1)
    # calc scores
    roc_auc = roc_auc_score(labels, logists)
    ap_score = average_precision_score(labels, logists)
    precisions, recalls, thresholds = precision_recall_curve(labels, logists)
    pr_auc = auc(recalls, precisions)
    warnings.simplefilter('ignore', RuntimeWarning)
    f1s = np.nan_to_num(2*precisions*recalls/(precisions+recalls))
    best_comb = np.argmax(f1s)
    f1 = f1s[best_comb]
    pre = precisions[best_comb]
    rec = recalls[best_comb]
    thresh = thresholds[best_comb]
    # calc reconstracted adj_mat and accuracy with the threshold for best f1
    features_rec = copy.deepcopy(Features_pred)
    features_rec[features_rec < thresh] = 0
    features_rec[features_rec >= thresh] = 1
    labels_all = torch.from_numpy(features.toarray()).view(-1)
    print(labels_all)
    preds_all = features_rec.view(-1)
    print(preds_all)
    recon_acc = (preds_all == labels_all).sum() / labels_all.size(0)
    results = {'roc': roc_auc,
               'pr': pr_auc,
               'ap': ap_score,
               'pre': pre,
               'rec': rec,
               'f1': f1,
               'acc': recon_acc,
               'adj_recon': features_rec}
    return results

def get_scores(edges_pos, edges_neg, A_pred, adj_label):
    # get logists and labels
    preds = A_pred[edges_pos.T]
    preds_neg = A_pred[edges_neg.T]
    logists = np.hstack([preds, preds_neg])
    labels = np.hstack([np.ones(preds.size(0)), np.zeros(preds_neg.size(0))])
    # logists = A_pred.view(-1)
    # labels = adj_label.to_dense().view(-1)
    # calc scores
    roc_auc = roc_auc_score(labels, logists)
    ap_score = average_precision_score(labels, logists)
    precisions, recalls, thresholds = precision_recall_curve(labels, logists)
    pr_auc = auc(recalls, precisions)
    warnings.simplefilter('ignore', RuntimeWarning)
    f1s = np.nan_to_num(2*precisions*recalls/(precisions+recalls))
    best_comb = np.argmax(f1s)
    f1 = f1s[best_comb]
    pre = precisions[best_comb]
    rec = recalls[best_comb]
    thresh = thresholds[best_comb]
    # calc reconstracted adj_mat and accuracy with the threshold for best f1
    adj_rec = copy.deepcopy(A_pred)
    adj_rec[adj_rec < thresh] = 0
    adj_rec[adj_rec >= thresh] = 1
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = adj_rec.view(-1).long()
    recon_acc = (preds_all == labels_all).sum().float() / labels_all.size(0)
    results = {'roc': roc_auc,
               'pr': pr_auc,
               'ap': ap_score,
               'pre': pre,
               'rec': rec,
               'f1': f1,
               'acc': recon_acc,
               'adj_recon': adj_rec}
    return results

def train_model(args, dl, vgae):
    optimizer = torch.optim.Adam(vgae.parameters(), lr=args.lr)
    # weights for log_lik loss
    adj_t = dl.adj_train
    #norm_w = adj_t.shape[0]**2 / float((adj_t.shape[0]**2 - adj_t.sum()) * 2)
    #TODO:看一下要不要加norm_w
    #pos_weight = torch.FloatTensor([float(adj_t.shape[0]**2 - adj_t.sum()) / adj_t.sum()]).to(args.device)
    # move input data and label to gpu if needed
    features = pickle.load(open(f'data/graphs/{args.dataset}_features.pkl', 'rb'))
    features_train,val_features,val_features_false,test_features,test_features_false=mask_test_features(features, args.val_frac, args.test_frac, args.no_mask)
    norm_w = features_train.shape[0]*features_train.shape[1] / float((features_train.shape[0]*features_train.shape[1] - features_train.sum()) * 2)
    pos_weight = torch.FloatTensor([float(features_train.shape[0]*features_train.shape[1] - features_train.sum()) / features_train.sum()]).to(args.device)
    if sp.issparse(features_train):  # 是否为稀疏矩阵
        features_train = torch.FloatTensor(features.toarray()).to(args.device)
    #features = dl.features.to(args.device)
    #adj_label = dl.adj_label.to_dense().to(args.device)
    #features_label =dl.features_orig.todense()
    #features_label = torch.from_numpy(features).to(args.device)
    best_vali_criterion = 0.0
    best_state_dict = None
    vgae.train()
    for epoch in range(args.epochs):
        t = time.time()
        Features_pred = vgae(features_train)
        optimizer.zero_grad()
        #loss = log_lik = F.binary_cross_entropy_with_logits(Features_pred, features_label, pos_weight=pos_weight)
        loss = log_lik = norm_w*F.binary_cross_entropy_with_logits(Features_pred, features_train)
        if not args.gae:
            kl_divergence = 0.5/Features_pred.size(0) * (1 + 2*vgae.logstd - vgae.mean**2 - torch.exp(2*vgae.logstd)).sum(1).mean()
            loss -= kl_divergence
        Features_pred = torch.sigmoid(Features_pred).detach().cpu()
        r = get_scores_features(val_features,val_features_false,Features_pred, features)
       
        #r = get_scores(dl.val_edges, dl.val_edges_false, A_pred, dl.adj_label)
        print('Epoch{:3}: train_loss: {:.4f} recon_acc: {:.4f} val_roc: {:.4f} val_ap: {:.4f} f1: {:.4f} time: {:.4f}'.format(epoch+1, loss.item(), r['acc'], r['roc'], r['ap'], r['f1'],time.time()-t))
        if r[args.criterion] > best_vali_criterion:
            best_vali_criterion = r[args.criterion]
            best_state_dict = copy.deepcopy(vgae.state_dict())
            # r_test = get_scores(dl.test_edges, dl.test_edges_false, A_pred, dl.adj_label)
            r_test = r
            print("          test_roc: {:.4f} test_ap: {:.4f} test_f1: {:.4f} test_recon_acc: {:.4f}".format(
                    r_test['roc'], r_test['ap'], r_test['f1'], r_test['acc']))
        loss.backward()
        optimizer.step()
    print('Done! final')
    print("Done! final results: test_roc: {:.4f} test_ap: {:.4f} test_f1: {:.4f} test_recon_acc: {:.4f}".format(r_test['roc'], r_test['ap'], r_test['f1'], r_test['acc']))

    #vgae.load_state_dict(best_state_dict)
    return vgae

def gen_graphs(args, dl, vgae):
    adj_orig = dl.adj_orig
    assert adj_orig.diagonal().sum() == 0
    # sp.csr_matrix
    if args.gae:
        pickle.dump(adj_orig, open(f'graphs/{args.dataset}_graph_0_gae.pkl', 'wb'))
    else:
        pickle.dump(adj_orig, open(f'graphs/{args.dataset}_graph_0.pkl', 'wb'))
    # sp.lil_matrix
    pickle.dump(dl.features_orig, open(f'graphs/{args.dataset}_features.pkl', 'wb'))
    features = dl.features.to(args.device)
    for i in range(args.gen_graphs):
        with torch.no_grad():
            A_pred = vgae(features)
        A_pred = torch.sigmoid(A_pred).detach().cpu()
        r = get_scores(dl.val_edges, dl.val_edges_false, A_pred, dl.adj_label)
        adj_recon = A_pred.numpy()
        np.fill_diagonal(adj_recon, 0)
        # np.ndarray
        if args.gae:
            filename = f'graphs/{args.dataset}_graph_{i+1}_logits_gae.pkl'
        else:
            filename = f'graphs/{args.dataset}_graph_{i+1}_logits.pkl'
        pickle.dump(adj_recon, open(filename, 'wb'))
