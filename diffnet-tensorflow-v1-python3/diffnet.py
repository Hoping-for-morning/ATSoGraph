""" diffnet.py 定义网络结构"""

# use tensorflow v1
import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()

import numpy as np


class diffnet():
    def __init__(self, conf):
        self.conf = conf
        self.supply_set = (
            'SOCIAL_NEIGHBORS_SPARSE_MATRIX',
            'CONSUMED_ITEMS_SPARSE_MATRIX'
        )

    def startConstructGraph(self):
        self.initializeNodes()
        self.createLoss()
        self.createOptimizer()
        self.createAdversarial()
        self.saveVariables()
        self.defineMap()

    # 输入和构造稀疏矩阵
    def inputSupply(self, data_dict):
        self.social_neighbors_indices_input = data_dict['SOCIAL_NEIGHBORS_INDICES_INPUT']
        self.social_neighbors_values_input = data_dict['SOCIAL_NEIGHBORS_VALUES_INPUT']

        self.consumed_items_indices_input = data_dict['CONSUMED_ITEMS_INDICES_INPUT']
        self.consumed_items_values_input = data_dict['CONSUMED_ITEMS_VALUES_INPUT']

        # prepare sparse matrix, in order to compute user's embedding from social neighbors and consumed items
        self.social_neighbors_dense_shape = np.array([self.conf.num_users, self.conf.num_users]).astype(np.int64)
        self.consumed_items_dense_shape = np.array([self.conf.num_users, self.conf.num_items]).astype(np.int64)

        self.social_neighbors_sparse_matrix = tf.compat.v1.SparseTensor(
            indices=self.social_neighbors_indices_input,
            values=self.social_neighbors_values_input,
            dense_shape=self.social_neighbors_dense_shape
        )
        self.consumed_items_sparse_matrix = tf.compat.v1.SparseTensor(
            indices=self.consumed_items_indices_input,
            values=self.consumed_items_values_input,
            dense_shape=self.consumed_items_dense_shape
        )

    """转换和嵌入生成"""
    # Conversion and Embedding Generation
    """基于均值和方差对输入x的分布进行归一化来转换该分布"""

    def convertDistribution(self, x):
        mean, var = tf.compat.v1.nn.moments(x, axes=[0, 1])
        y = (x - mean) * 0.2 / tf.compat.v1.sqrt(var)
        return y

    """使用稀疏矩阵乘法计算来自社交邻居的用户嵌入"""

    def generateUserEmbeddingFromSocialNeighbors(self, current_user_embedding):
        user_embedding_from_social_neighbors = tf.compat.v1.sparse_tensor_dense_matmul(
            self.social_neighbors_sparse_matrix, current_user_embedding
        )
        return user_embedding_from_social_neighbors

    """使用稀疏矩阵乘法计算消费项目的用户嵌入"""

    def generateUserEmebddingFromConsumedItems(self, current_item_embedding):
        user_embedding_from_consumed_items = tf.compat.v1.sparse_tensor_dense_matmul(
            self.consumed_items_sparse_matrix, current_item_embedding
        )
        return user_embedding_from_consumed_items

    """
    初始化节点
    包括：
    item_input
    user_input
    label_input
    the embedding of user and item
    review_vector_matrix 评审向量矩阵 代表了 feature embedding 的表示
    fusion_layer 融合层
    """

    def initializeNodes(self):
        self.item_input = tf.compat.v1.placeholder("int32", [None, 1])  # Get item embedding from the core_item_input
        self.user_input = tf.compat.v1.placeholder("int32", [None, 1])  # Get user embedding from the core_user_input
        self.labels_input = tf.compat.v1.placeholder("float32", [None, 1])

        self.user_embedding = tf.compat.v1.Variable(
            tf.compat.v1.random_normal([self.conf.num_users, self.conf.dimension], stddev=0.01), name='user_embedding')
        self.item_embedding = tf.compat.v1.Variable(
            tf.compat.v1.random_normal([self.conf.num_items, self.conf.dimension], stddev=0.01), name='item_embedding')

        # perturbation variables
        # user
        self.delta_u = tf.compat.v1.Variable(
            tf.compat.v1.zeros(shape=[self.conf.num_users, self.conf.dimension], name='delta_u')
        )
        # item
        self.delta_i = tf.compat.v1.Variable(
            tf.compat.v1.zeros(shape=[self.conf.num_items, self.conf.dimension], name='delta_i')
        )

        self.user_review_vector_matrix = tf.compat.v1.constant( \
            np.load(self.conf.user_review_vector_matrix), dtype=tf.compat.v1.float32)
        self.item_review_vector_matrix = tf.compat.v1.constant( \
            np.load(self.conf.item_review_vector_matrix), dtype=tf.compat.v1.float32)
        self.reduce_dimension_layer = tf.compat.v1.layers.Dense( \
            self.conf.dimension, activation=tf.compat.v1.nn.sigmoid, name='reduce_dimension_layer')

        self.item_fusion_layer = tf.compat.v1.layers.Dense( \
            self.conf.dimension, activation=tf.compat.v1.nn.sigmoid, name='item_fusion_layer')
        self.user_fusion_layer = tf.compat.v1.layers.Dense( \
            self.conf.dimension, activation=tf.compat.v1.nn.sigmoid, name='user_fusion_layer')
        self.init = tf.compat.v1.global_variables_initializer()

    """构造训练图"""
    def createInference(self):
        """
        嵌入降维
        通过转换分布，降维的方式，映射原来的 review matrix
        """
        # handle review information, map the origin review into the new space and
        first_user_review_vector_matrix = self.convertDistribution(self.user_review_vector_matrix)
        first_item_review_vector_matrix = self.convertDistribution(self.item_review_vector_matrix)

        self.user_reduce_dim_vector_matrix = self.reduce_dimension_layer(first_user_review_vector_matrix)
        self.item_reduce_dim_vector_matrix = self.reduce_dimension_layer(first_item_review_vector_matrix)

        second_user_review_vector_matrix = self.convertDistribution(self.user_reduce_dim_vector_matrix)
        second_item_review_vector_matrix = self.convertDistribution(self.item_reduce_dim_vector_matrix)

        """项目嵌入是通过将 item embedding 和 item_review_vector_matrix 进行融合来计算的"""
        # compute item embedding
        # self.fusion_item_embedding = self.item_fusion_layer(\
        #   tf.compat.v1.concat([self.item_embedding, second_item_review_vector_matrix], 1))

        # fusion item embedding
        self.fusion_item_embedding = self.item_embedding + second_item_review_vector_matrix
        self.final_item_embedding = self.fusion_item_embedding
        # self.final_item_embedding = self.fusion_item_embedding = second_item_review_vector_matrix

        # fusion user embedding
        self.fusion_user_embedding = self.user_embedding + second_user_review_vector_matrix

        """两层图卷积"""
        first_gcn_user_embedding = self.generateUserEmbeddingFromSocialNeighbors(self.fusion_user_embedding)
        second_gcn_user_embedding = self.generateUserEmbeddingFromSocialNeighbors(first_gcn_user_embedding)

        # ORIGINAL OPERATION OF diffnet
        # self.final_user_embedding = second_gcn_user_embedding + user_embedding_from_consumed_items

        """最终的 user embedding 需要加上来自项目消费的嵌入"""
        #history consume item embedding
        user_embedding_from_consumed_items = self.generateUserEmebddingFromConsumedItems(self.final_item_embedding)

        # FOLLOWING OPERATION IS USED TO TACKLE THE GRAPH OVERSMOOTHING ISSUE, IF YOU WANT TO KNOW MORE DETAILS, PLEASE REFER TO https://github.com/newlei/LR-GCCF
        self.final_user_embedding = first_gcn_user_embedding + second_gcn_user_embedding + user_embedding_from_consumed_items

        """取出对应的 index 作为隐向量"""
        latest_user_latent = tf.compat.v1.gather_nd(self.final_user_embedding, self.user_input)
        latest_item_latent = tf.compat.v1.gather_nd(self.final_item_embedding, self.item_input)

        predict_vector = tf.compat.v1.multiply(latest_user_latent, latest_item_latent)

        """sigmoid 函数最终预测"""
        return tf.compat.v1.sigmoid(tf.compat.v1.reduce_sum(predict_vector, 1, keepdims=True))
        # self.prediction = self.predict_rating_layer(tf.compat.v1.concat([latest_user_latent, latest_item_latent], 1))

    def createInference_adv(self):
        """
        嵌入降维
        通过转换分布，降维的方式，映射原来的 review matrix
        """
        # handle review information, map the origin review into the new space and
        first_user_review_vector_matrix = self.convertDistribution(self.user_review_vector_matrix)
        first_item_review_vector_matrix = self.convertDistribution(self.item_review_vector_matrix)

        self.user_reduce_dim_vector_matrix = self.reduce_dimension_layer(first_user_review_vector_matrix)
        self.item_reduce_dim_vector_matrix = self.reduce_dimension_layer(first_item_review_vector_matrix)

        second_user_review_vector_matrix = self.convertDistribution(self.user_reduce_dim_vector_matrix)
        second_item_review_vector_matrix = self.convertDistribution(self.item_reduce_dim_vector_matrix)

        """项目嵌入是通过将 item embedding 和 item_review_vector_matrix 进行融合来计算的"""
        # compute item embedding
        # self.fusion_item_embedding = self.item_fusion_layer(\
        #   tf.compat.v1.concat([self.item_embedding, second_item_review_vector_matrix], 1))

        # fusion item embedding
        self.fusion_item_embedding = self.item_embedding + second_item_review_vector_matrix
        self.final_item_embedding = self.fusion_item_embedding
        # self.final_item_embedding = self.fusion_item_embedding = second_item_review_vector_matrix

        # fusion user embedding
        # self.fusion_user_embedding = self.user_fusion_layer(\
        #    tf.compat.v1.concat([self.user_embedding, second_user_review_vector_matrix], 1))
        self.fusion_user_embedding = self.user_embedding + second_user_review_vector_matrix

        """ 引入干扰，这里很关键，可能会有问题 """
        """ 等下把公式补上 """
        # add adversrial noise
        # user
        self.fusion_user_embedding = self.fusion_user_embedding \
                                     + tf.compat.v1.reduce_sum(
            tf.compat.v1.gather_nd(self.delta_i, self.user_input))
        # item
        self.final_item_embedding = self.final_item_embedding \
                                    + tf.compat.v1.reduce_sum(
            tf.compat.v1.gather_nd(self.delta_i, self.item_input))

        """两层图卷积"""
        first_gcn_user_embedding = self.generateUserEmbeddingFromSocialNeighbors(self.fusion_user_embedding)
        second_gcn_user_embedding = self.generateUserEmbeddingFromSocialNeighbors(first_gcn_user_embedding)

        # ORIGINAL OPERATION OF diffnet
        # self.final_user_embedding = second_gcn_user_embedding + user_embedding_from_consumed_items

        """最终的 user embedding 需要加上来自项目消费的嵌入"""
        # history consume item embedding
        user_embedding_from_consumed_items = self.generateUserEmebddingFromConsumedItems(self.final_item_embedding)

        # FOLLOWING OPERATION IS USED TO TACKLE THE GRAPH OVERSMOOTHING ISSUE, IF YOU WANT TO KNOW MORE DETAILS, PLEASE REFER TO https://github.com/newlei/LR-GCCF
        self.final_user_embedding = first_gcn_user_embedding + second_gcn_user_embedding + user_embedding_from_consumed_items

        """取出对应的 index 作为隐向量"""
        latest_user_latent = tf.compat.v1.gather_nd(self.final_user_embedding, self.user_input)
        latest_item_latent = tf.compat.v1.gather_nd(self.final_item_embedding, self.item_input)

        predict_vector = tf.compat.v1.multiply(latest_user_latent, latest_item_latent)

        """sigmoid 函数最终预测"""
        return tf.compat.v1.sigmoid(tf.compat.v1.reduce_sum(predict_vector, 1, keepdims=True))
        # self.prediction = self.predict_rating_layer(tf.compat.v1.concat([latest_user_latent, latest_item_latent], 1))

    def createLoss(self):
        """损失为预测值和标签之间的l2损失"""
        # loss for L(Theta)
        self.prediction = self.createInference()
        self.loss = tf.compat.v1.nn.l2_loss(self.labels_input - self.prediction)

        # loss for adv
        self.prediction_adv = self.createInference_adv()
        self.loss_adv = tf.compat.v1.nn.l2_loss(self.labels_input - self.prediction_adv)

        self.opt_loss = self.loss + self.loss_adv

    def createOptimizer(self):
        """优化器为Adam"""
        self.opt = tf.compat.v1.train.AdamOptimizer(self.conf.learning_rate).minimize(self.opt_loss)

    def createAdversarial(self):
        """---------------------------------Add adversarial training---------------------------------------"""
        """ 应该在 embedding 之后构造 perturbation """
        """ PGD 方法加入扰动 in fusion embedding """
        # generate the adversarial weights by gradient-based method
        # return the IndexedSlice Data: [(values, indices, dense_shape)]
        # grad_var_P: [grad,var], grad_var_Q: [grad, var]
        # grad_u : for user
        # grad_i : for item
        """random"""
        # generation
        self.adv_P = tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01)
        self.adv_Q = tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01)

        # normalization and multiply epsilon
        self.update_u = self.delta_u.assign(tf.nn.l2_normalize(self.adv_P, 1) * self.eps)
        self.update_i = self.delta_i.assign(tf.nn.l2_normalize(self.adv_Q, 1) * self.eps)


        """grad"""
        # self.grad_u, self.grad_i = tf.compat.v1.gradients(self.loss,
        #                                                   [self.fusion_user_embedding, self.final_item_embedding])
        #
        # # convert the IndexedSlice Data to Dense Tensor
        # self.grad_u_dense = tf.compat.v1.stop_gradient(self.grad_u)
        # self.grad_i_dense = tf.compat.v1.stop_gradient(self.grad_i)
        #
        # # normalization: new_grad = (grad / |grad|) * eps
        # self.update_u = self.delta_u.assign(tf.nn.l2_normalize(self.grad_u_dense, 1) * self.eps)
        # self.update_i = self.delta_i.assign(tf.nn.l2_normalize(self.grad_i_dense, 1) * self.eps)



    """保存变量，创建一个变量字典，保存 user and item embedding， 使用 tf.saver 保存变量"""

    def saveVariables(self):
        ############################# Save Variables #################################
        variables_dict = {}
        variables_dict[self.user_embedding.op.name] = self.user_embedding
        variables_dict[self.item_embedding.op.name] = self.item_embedding

        for v in self.reduce_dimension_layer.variables:
            variables_dict[v.op.name] = v

        self.saver = tf.compat.v1.train.Saver(variables_dict)
        ############################# Save Variables #################################

    """定义 map 数据结构，完成了输入映射"""

    def defineMap(self):
        map_dict = {}
        map_dict['train'] = {
            self.user_input: 'USER_LIST',
            self.item_input: 'ITEM_LIST',
            self.labels_input: 'LABEL_LIST'
        }

        map_dict['val'] = {
            self.user_input: 'USER_LIST',
            self.item_input: 'ITEM_LIST',
            self.labels_input: 'LABEL_LIST'
        }

        map_dict['test'] = {
            self.user_input: 'USER_LIST',
            self.item_input: 'ITEM_LIST',
            self.labels_input: 'LABEL_LIST'
        }

        map_dict['eva'] = {
            self.user_input: 'EVA_USER_LIST',
            self.item_input: 'EVA_ITEM_LIST'
        }

        map_dict['out'] = {
            'train': self.loss,
            'val': self.loss,
            'test': self.loss,
            'eva': self.prediction
        }

        self.map_dict = map_dict
