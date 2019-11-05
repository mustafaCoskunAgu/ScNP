clear;
clc;
%This is the inverse of Cora Normalized matrix
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