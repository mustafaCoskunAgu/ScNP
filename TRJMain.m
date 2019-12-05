clear;
clc;
% First Run this code and save the file 
%Then go to https://github.com/klicperajo/ppnp
%And paste below code into propagation.py given by Klicpera et al
% class PPRExactCorr:
%     def __init__(self, adj_matrix: sp.spmatrix, alpha: float):
%         self.alpha = alpha
%         mat = hd.loadmat('Cora4.mat')
%         #mat = sio.loadmat('Cite5.mat')
%         prop_appnpC = mat['prop_ppnpC']  
%         self.ppr_mat = prop_appnpC
% 
%     def build_model(self, Z: tf.Tensor, keep_prob: float) -> tf.Tensor:
%         with tf.variable_scope(f'Propagation'):
%             ppr_mat_tf = tf.constant(self.ppr_mat, dtype=tf.float32)
%             ppr_drop = tf.nn.dropout(ppr_mat_tf, keep_prob)
%             return ppr_drop @ Z
%PPRMatrix is the inverse of Cora Normalized matrix
%Note: You can either compute this inverse by Coskun et al, 
%@inproceedings{CoskunKdd,
% title={Efficient processing of network proximity queries via chebyshev acceleration},
%  author={Coskun, Mustafa and Grama, Ananth and Koyuturk, Mehmet},
%  booktitle={Proceedings of the 22Nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
%  pages={1515--1524},
%  year={2016},
%  organization={ACM}
%}
%or
%@article{CoskunVldb,
% title={Indexed fast network proximity querying},
%  author={Co{\c{s}}kun, Mustafa and Grama, Ananth and Koyut{\"u}rk, Mehmet},
%  journal={Proceedings of the VLDB Endowment},
%  volume={11},
%  number={8},
%  pages={840--852},
%  year={2018},
%  publisher={VLDB Endowment}
%}
%Or small matrices you can use inv(I-\alpha*P) in matlab or 
%You can use following function in python:
%def calc_A_hat(adj_matrix: sp.spmatrix) -> sp.spmatrix:
%    nnodes = adj_matrix.shape[0]
%    A = adj_matrix + sp.eye(nnodes)
%    D_vec = np.sum(A, axis=1).A1
%    D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
%    D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)
%    return D_invsqrt_corr @ A @ D_invsqrt_corr
%def calc_ppr_exact(adj_matrix: sp.spmatrix, alpha: float) -> np.ndarray:
%    nnodes = adj_matrix.shape[0]
%    M = calc_A_hat(adj_matrix)
%    A_inner = sp.eye(nnodes) - (1 - alpha) * M
%    return alpha * np.linalg.inv(A_inner.toarray())
load('PPMatrix.mat');
PPRMatrix = prop_ppnp.ppr_mat;

[n,~] = size(PPRMatrix);


CorrMatrix2 = zeros(n,n);
newRuntime = 0;
tic;
    for i = 1:n
      for j = 1:i
        if(i == j)
           %Do notting
        else
        [I1,~] =find(PPRMatrix(:,i)<=1e-4);
        [I2,~] =find(PPRMatrix(:,j)<=1e-4);
        ind = intersect(I1,I2);
        vec = [PPRMatrix(:,i), PPRMatrix(:,j)];
        vec(ind, :) = [];
        CorrMatrix2(i, j) = corr(vec(:,1),vec(:,2));
        end


      end
    end

newRuntime = toc+ newRuntime;
fprintf('Time %f\n',newRuntime);





 clear PPNRMatrix;
 prop_ppnpC = CorrMatrix2 + CorrMatrix2' + eye(n);
 save('Cora4.mat', 'prop_ppnpC', '-v7.3');
