import torch
import numpy as np


def sim(z_i, z_j):
    """Normalized dot product between two vectors.

    Inputs:
    - z_i: 1xD tensor.
    - z_j: 1xD tensor.
    
    Returns:
    - A scalar value that is the normalized dot product between z_i and z_j.
    """
    norm_dot_product = None
    ##############################################################################
    # TODO: Start of your code.                                                  #
    #                                                                            #
    # HINT: torch.linalg.norm might be helpful.                                  #
    ##############################################################################

    # 计算两个向量的点积
    dot_product = torch.sum(z_i * z_j)
    # 计算向量z_i的范数
    norm_i = torch.linalg.norm(z_i)
    # 计算向量z_j的范数
    norm_j = torch.linalg.norm(z_j)
    # 计算归一化点积，即点积除以两个向量范数的乘积
    norm_dot_product = dot_product / (norm_i * norm_j)
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    
    return norm_dot_product


def simclr_loss_naive(out_left, out_right, tau):
    """Compute the contrastive loss L over a batch (naive loop version).
    
    Input:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch. The same row in out_left and out_right form a positive pair. 
    In other words, (out_left[k], out_right[k]) form a positive pair for all k=0...N-1.
    - tau: scalar value, temperature parameter that determines how fast the exponential increases.
    
    Returns:
    - A scalar value; the total loss across all positive pairs in the batch. See notebook for definition.
    """
    N = out_left.shape[0]  # total number of training examples
    
     # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    total_loss = 0
    for k in range(N):  # loop through each positive pair (k, k+N)
        z_k, z_k_N = out[k], out[k+N]
        
        ##############################################################################
        # TODO: Start of your code.                                                  #
        #                                                                            #
        # Hint: Compute l(k, k+N) and l(k+N, k).                                     #
        ##############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # 计算 l(k, k+N)
        left_numerator = (sim(z_k, z_k_N) / tau).exp()
        left_need_sim = out[np.arange(2 * N) != k]
        left_denominator = torch.tensor([sim(z_k, z_i) / tau for z_i in left_need_sim]).exp().sum()
        left = -(left_numerator / left_denominator).log()

        # 计算 l(k+N, k)
        right_numerator = (sim(z_k_N, z_k) / tau).exp()
        right_need_sim = out[np.arange(2 * N) != k + N]
        right_denominator = torch.tensor([sim(z_k_N, z_i) / tau for z_i in right_need_sim]).exp().sum()
        right = -(right_numerator / right_denominator).log()

        total_loss += left + right
        
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
         ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
    
    # In the end, we need to divide the total loss by 2N, the number of samples in the batch.
    total_loss = total_loss / (2*N)
    return total_loss


def sim_positive_pairs(out_left, out_right):
    """Normalized dot product between positive pairs.

    Inputs:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch.
    The same row in out_left and out_right form a positive pair.
    
    Returns:
    - A Nx1 tensor; each row k is the normalized dot product between out_left[k] and out_right[k].
    """
    pos_pairs = None
    
    ##############################################################################
    # TODO: Start of your code.                                                  #
    #                                                                            #
    # HINT: torch.linalg.norm might be helpful.                                  #
    ##############################################################################
    
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 获取批次大小（也就是正样本对的数量）
    N = out_left.shape[0]
    # 初始化一个空的Nx1张量，用于存储正样本对的归一化点积
    pos_pairs = torch.zeros((N, 1))
    for k in range(N):
        # 获取第k对正样本对应的向量
        z_k_left = out_left[k]
        z_k_right = out_right[k]
        # 计算点积
        dot_product = torch.sum(z_k_left * z_k_right)
        # 计算左向量的范数
        norm_left = torch.linalg.norm(z_k_left)
        # 计算右向量的范数
        norm_right = torch.linalg.norm(z_k_right)
        # 计算归一化点积
        norm_dot_product = dot_product / (norm_left * norm_right)
        # 将第k对正样本的归一化点积存入结果张量中
        pos_pairs[k, 0] = norm_dot_product

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return pos_pairs


def compute_sim_matrix(out):
    """Compute a 2N x 2N matrix of normalized dot products between all pairs of augmented examples in a batch.

    Inputs:
    - out: 2N x D tensor; each row is the z-vector (output of projection head) of a single augmented example.
    There are a total of 2N augmented examples in the batch.
    
    Returns:
    - sim_matrix: 2N x 2N tensor; each element i, j in the matrix is the normalized dot product between out[i] and out[j].
    """
    sim_matrix = None
    
    ##############################################################################
    # TODO: Start of your code.                                                  #
    ##############################################################################
    
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 获取输入张量的行数（也就是增强样本的数量），应为2N
    num_samples = out.shape[0]
    # 初始化一个2N x 2N的全零张量，用于存储相似度矩阵
    sim_matrix = torch.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(num_samples):
            # 获取第i个样本的特征向量
            sample_i = out[i]
            # 获取第j个样本的特征向量
            sample_j = out[j]
            # 计算点积
            dot_product = torch.sum(sample_i * sample_j)
            # 计算第i个样本特征向量的范数
            norm_i = torch.linalg.norm(sample_i)
            # 计算第j个样本特征向量的范数
            norm_j = torch.linalg.norm(sample_j)
            # 计算归一化点积
            norm_dot_product = dot_product / (norm_i * norm_j)
            # 将归一化点积结果存入相似度矩阵相应位置
            sim_matrix[i, j] = norm_dot_product

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return sim_matrix


def simclr_loss_vectorized(out_left, out_right, tau, device='cuda'):
    """Compute the contrastive loss L over a batch (vectorized version). No loops are allowed.
    
    Inputs and output are the same as in simclr_loss_naive.
    """
    N = out_left.shape[0]
    
    # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    # Compute similarity matrix between all pairs of augmented examples in the batch.
    sim_matrix = compute_sim_matrix(out)  # [2*N, 2*N]
    
    ##############################################################################
    # TODO: Start of your code. Follow the hints.                                #
    ##############################################################################
    
    # Step 1: Use sim_matrix to compute the denominator value for all augmented samples.
    # Hint: Compute e^{sim / tau} and store into exponential, which should have shape 2N x 2N.
    exponential = (sim_matrix / tau).exp().to(device)
    
    # This binary mask zeros out terms where k=i.
    mask = (torch.ones_like(exponential, device=device) - torch.eye(2 * N, device=device)).to(device).bool()
    
    # We apply the binary mask.
    exponential = exponential.masked_select(mask).view(2 * N, -1)  # [2*N, 2*N-1]
    
    # Hint: Compute the denominator values for all augmented samples. This should be a 2N x 1 vector.
    denom = exponential.sum(dim=1)


    # Step 2: Compute similarity between positive pairs.
    # You can do this in two ways: 
    # Option 1: Extract the corresponding indices from sim_matrix. 
    # Option 2: Use sim_positive_pairs().
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    sim_pairs = sim_positive_pairs(out_left, out_right).to(device)
    sim_pairs = torch.cat([sim_pairs, sim_pairs], dim=0)

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Step 3: Compute the numerator value for all augmented samples.
    numerator = (sim_pairs / tau).exp()
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Step 4: Now that you have the numerator and denominator for all augmented samples, compute the total loss.
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    loss = torch.mean(-torch.log(numerator / denom))
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    
    return loss


def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))