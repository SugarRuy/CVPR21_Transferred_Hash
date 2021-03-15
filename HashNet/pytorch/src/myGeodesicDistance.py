import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable

from myGetAdvVulnerable import get_test_dis, choose_index_by_dis_method
import matplotlib.pyplot as plt

def get_target_id_matrix(test_true_label_y, i_index_set, j_index_matrix):
    i_max = i_index_set.shape[0]
    j_max = j_index_matrix.shape[1]
    target_id_mat = np.zeros([i_max, j_max])
    for i in range(i_max):
        target_id_mat[i] = np.array([j_index_matrix[int(test_true_label_y[i_index_set[i]])]]).astype(int)
    return target_id_mat

def get_source_id_array(test_true_id_x, i_index_set):
    i_max = i_index_set.shape[0]
    source_id_array = np.array([(test_true_id_x[i_index_set[i]]) for i in range(i_max)]).astype(int)
    return source_id_array

def get_geodesic_dist_mat(code_test, code, test_true_id_x, test_true_label_y, i_index_set, j_index_matrix, knn_k=50, isTargetClasses=True, geodesic_method='hopping'):
    # if isTargetClasses, then the target could be any code in the class
    # if not isTargetClasses, then the target must be the exact same code.
    # the target code will be added into the test set code, combining to a new code mat,
    # which is used to calculated the distance.
    i_max = i_index_set.shape[0]
    j_max = j_index_matrix.shape[1]

    source_id_array = get_source_id_array(test_true_id_x, i_index_set)
    target_id_mat = get_target_id_matrix(test_true_label_y, i_index_set, j_index_matrix).astype(int)
    if not isTargetClasses:
        target_id_mat = reMapping_target_id_mat(target_id_mat, code=code, code_test=code_test).astype(int)
        hashcode = code_test
    else:
        hashcode = code_test

    hamming_dist_mat = get_hamming_dist_matrix(hashcode)
    kNN_mat = cal_knn_matrix(hamming_dist_mat, k=knn_k)

    # create csgraph for using dijkstra
    if geodesic_method == 'isomap':
        max_distance = 999999
        csgraph = np.zeros_like(hamming_dist_mat)+max_distance
        for i in range(kNN_mat.shape[0]):
            for j in range(kNN_mat.shape[1]):
                csgraph[i][int(kNN_mat[i,j])] = hamming_dist_mat[i][int(kNN_mat[i,j])]
        from scipy.sparse.csgraph import dijkstra
        dist_matrix = dijkstra(csgraph, indices=source_id_array)


    geodesic_dist_mat = np.zeros([i_max, j_max])
    for i in range(i_max):
        for j in range(j_max):
            if geodesic_method == 'hopping':
                geodesic_dist_mat[i, j] = cal_geodesic_dist(kNN_mat, source_id_array[i], target_id_mat[i, j])
            elif geodesic_method == 'isomap':
                geodesic_dist_mat[i, j] = dist_matrix[i][target_id_mat[i, j]]
    return geodesic_dist_mat


def reMapping_target_id_mat(target_id_mat, code, code_test):
    size_i, size_j = target_id_mat.shape
    reMapped_target_id_mat = np.zeros([size_i, size_j])
    hashbit = code.shape[1]

    for i in range(size_i):
        target_i_code = code[target_id_mat[i]]
        hamming_dist_i = np.matmul(target_i_code, code_test.transpose()) * (-0.5) + hashbit / 2
        arg_min_id = np.argmin(hamming_dist_i, axis=1)
        reMapped_target_id_mat[i] = arg_min_id
    return reMapped_target_id_mat


def get_hamming_dist_matrix(hashcode):
    #
    data_size = hashcode.shape[0]
    hashbit = hashcode.shape[1]
    hamming_dist_mat = np.matmul(hashcode, hashcode.transpose()) * (-0.5) + hashbit / 2
    return hamming_dist_mat


def cal_knn_matrix(hamming_dist_mat, k=10):
    data_size = hamming_dist_mat.shape[0]
    kNN_mat = np.zeros([data_size, k])
    for i in range(data_size):
        # np.argsort(hamming_dist_mat[i])[:k]
        kNN_mat[i] = np.argpartition(hamming_dist_mat[i], k)[:k]

    return kNN_mat.astype(int)

def cal_isomap_dist(kNN_mat, hamming_dist_mat, source_id, target_id):

    return

def cal_geodesic_dist(kNN_mat, source_id, target_id):

    # use BFS search
    target_id = int(target_id)
    data_size = kNN_mat.shape[0]
    k = kNN_mat.shape[1]
    depth = 0
    # 0 for white, 1 for gray, 2 for black
    flag_array = np.zeros([data_size])
    path_array = np.zeros([data_size]) - 1
    import queue
    queue = queue.Queue(data_size)
    queue.put(source_id)
    # set the gray
    flag_array[source_id] = 1
    bFindIt = False

    while not queue.empty():
        # push from queue, set to black
        i = int(queue.get())
        flag_array[source_id] = 2

        for neighbor_id in kNN_mat[i]:
            # if not visited, else skip it
            neighbor_id = int(neighbor_id)
            if flag_array[neighbor_id] == 0:
                # set to gray and set the path
                flag_array[neighbor_id] = 1
                path_array[neighbor_id] = i
                #if multi_label_test[neighbor_id] == database_label[target_id]:
                if neighbor_id == target_id:
                    # the target is founded, clean the queue and break the loop
                    while not queue.empty():
                        queue.get()
                    bFindIt = True
                    break

                else:
                    queue.put(neighbor_id)
                    flag_array[neighbor_id] = 1
    if not bFindIt:
        return data_size + 1

    result_id = int(neighbor_id)

    print("Result ID(same class as target id):%d" % (result_id))
    while path_array[result_id] != source_id:
        depth += 1
        result_id = int(path_array[result_id])
        print("child of:%d" % (result_id))
    print("Source ID:%d" % (source_id))
    return depth


'''
The following segment is used for computing the geodesic distance for an adv to its approximated manifold. 
'''


def cal_knn_index_pixel(adv_img, imgs_test, k=50):
    # calculate the k nearest neighbors' index in the inputs level.
    # return the kNN results in the test set
    l2_distance_adv = np.zeros([imgs_test.shape[0]])
    for i in range(imgs_test.shape[0]):
        l2_distance_adv[i] = np.linalg.norm((imgs_test[i] - adv_img).reshape([-1]), ord=2, axis=-1)
    #l2_distance_adv = np.linalg.norm((imgs_test[0]-adv_img).reshape([imgs_test.shape[0], -1]), ord=2, axis=-1)
    # k nearest neighbors' indexs. The order of the indexs is not defined.
    kNN_indexs = np.argpartition(l2_distance_adv, k)[:k]
    return kNN_indexs

def get_kNN_imgs(input_img, imgs_manifold, k):
    kNN_indexes = cal_knn_index_pixel(input_img, imgs_manifold, k=k)
    return imgs_manifold[kNN_indexes]


def optimize_least_squared_beta(adv_ori_diff, X_mat):
    # get the optimized beta parameter
    beta_optimzed, _,_,_ = np.linalg.lstsq(X_mat, adv_ori_diff)
    return beta_optimzed


def cal_geo_dist_adv(ori_img, adv_img, imgs_manifold, k=50):
    # calculate geodesic distance between adv and its approximated manifold
    kNN_imgs = get_kNN_imgs(adv_img, imgs_manifold, k=k)
    adv_ori_diff = (adv_img - ori_img).reshape([-1])
    X_mat = (kNN_imgs - kNN_imgs.mean(axis=0)).reshape([kNN_imgs.shape[0], -1]).transpose()

    #beta_optimzed = optimize_least_squared_beta(adv_ori_diff, X_mat)
    beta_optimzed = optimize_least_squared_beta(adv_ori_diff.reshape([-1]), X_mat)

    projected_adv = np.matmul(X_mat, beta_optimzed)
    #geo_dist_adv = np.linalg.norm(projected_adv, ord=2)
    geo_dist_adv = project_orthogonal(adv_ori_diff, X_mat)
    #print np.linalg.norm(adv_img.reshape([-1]), ord=2)
    print(geo_dist_adv)
    return geo_dist_adv


def project_orthogonal(basis, vectors, rank=None):
    """
    Project the given vectors on the basis using an orthogonal projection.
    :param basis: basis vectors to project on
    :type basis: numpy.ndarray
    :param vectors: vectors to project
    :type vectors: numpy.ndarray
    :return: projection
    :rtype: numpy.ndarray
    """

    # The columns of Q are an orthonormal basis of the columns of basis
    Q, R = np.linalg.qr(basis)
    if rank is not None and rank > 0:
        Q = Q[:, :rank]

    # As Q is orthogonal, the projection is
    beta = Q.T.dot(vectors)
    projection = Q.dot(beta)

    return projection


def cal_geo_dist_github(ori_img, adv_img, imgs_manifold, k=50, return_projections=False, kNN_indexes = None):
    '''

    Args:
        ori_img: The original image
        adv_img: The adversarial image
        imgs_manifold: The images manifold to project
        k: k of kNN
        return_projections: Flag to decided if returns projections and the mean of sub-manifold
        kNN_indexes: The indexes of images on the sub-manifold. If set to None, then using kNN to compute the sub-manifold

    Returns:
        distances: The distances from images to their sub-manifold projections
        [return2]: The projections on sub-manifold
        nearest_neighbor_mean: The mean value of the images on sub-manifold.
    '''
    from myGeodesicDistance import cal_geo_dist_github
    if kNN_indexes is None:
        kNN_indexes = cal_knn_index_pixel(adv_img, imgs_manifold, k=k)
    else:
        k = kNN_indexes.shape[0]
    kNN_imgs = imgs_manifold[kNN_indexes]
    #kNN_imgs = get_kNN_imgs(adv_img, imgs_manifold, k=k)
    adv_ori_diff = (adv_img - ori_img).reshape([-1])
    X_mat = (kNN_imgs - kNN_imgs.mean(axis=0)).reshape([kNN_imgs.shape[0], -1]).transpose()

    perturbation = adv_img
    nearest_neighbor_mean = kNN_imgs.mean(axis=0)

    pure_perturbations = (perturbation - ori_img).reshape([-1])
    relative_perturbation = (perturbation - nearest_neighbor_mean).reshape([-1])
    relative_test_image = (ori_img - nearest_neighbor_mean).reshape([-1])
    basis = X_mat
    nearest_neighbor_vectors = np.stack((
        pure_perturbations,
        relative_perturbation,
        relative_test_image
    ), axis=1)
    nearest_neighbor_projections = project_orthogonal(basis, nearest_neighbor_vectors)
    distances = np.linalg.norm(nearest_neighbor_vectors - nearest_neighbor_projections, ord=2, axis=0)
    #print kNN_indexes
    print(np.linalg.norm(pure_perturbations-relative_perturbation, ord=2, axis=0))
    print(distances)
    if return_projections:
        #return distances, nearest_neighbor_projections[:,0].reshape(ori_img.shape)+ori_img
        #return distances, nearest_neighbor_projections[:, 1].reshape(ori_img.shape)+nearest_neighbor_mean, nearest_neighbor_mean
        return distances, nearest_neighbor_projections[:,0].reshape(ori_img.shape)+ori_img, nearest_neighbor_mean
    else:
        return distances


def get_projection(ori_img, adv_img, target_img, imgs_manifold, k=50, sub_manifold_method='adv'):
    if sub_manifold_method == 'adv':
        kNN_indexes = cal_knn_index_pixel(adv_img, imgs_manifold, k=k)
    elif sub_manifold_method == 'ori':
        kNN_indexes = cal_knn_index_pixel(ori_img, imgs_manifold, k=k)
    elif sub_manifold_method == 'target':
        kNN_indexes = cal_knn_index_pixel(target_img, imgs_manifold, k=k)

    _, projected_img, _ = cal_geo_dist_github(ori_img, adv_img, imgs_manifold, k=25, return_projections=True,
                                                          kNN_indexes=kNN_indexes)
    return projected_img


def fast_idea_verification(ori_img, adv_img, target_img, imgs_manifold, k=50):
    # This might be a bad idea, but it is worth a try
    from .publicFunctions import load_net_inputs, load_net_params, load_dset_params
    from .myRetrieval import get_img_num_by_class_from_img
    model1, snapshot_path, query_path, database_path = load_net_params(net1)

    get_img_num_by_class_from_img()
    return





def main_func():
    from .publicFunctions import load_net_inputs, load_net_params, load_dset_params
    npy_name = '/%s_imgs_step%1.1f_linf%d_%dx%d_%s.npy' % (adv_method, step, linf, i_max, j_max, dis_method)
    npy_path = 'save_for_load/' + net1 + npy_name

    path_white_test_dis_npy = 'save_for_load/distanceADVRetrieval/test_dis_%s.npy'%(net1)
    path_black_test_dis_npy = 'save_for_load/distanceADVRetrieval/test_dis_%s.npy'%(net2)
    dset_test, dset_database = load_dset_params(job_dataset)
    model1, snapshot_path, query_path, database_path = load_net_params(net1)
    tmp = np.load(database_path)
    _, code, multi_label = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']

    test_dis_white, test_dis_black = get_test_dis(path_white_test_dis_npy, path_black_test_dis_npy)
    test_true_id_x, test_true_label_y = choose_index_by_dis_method(dis_method, test_dis_white, test_dis_black,
                                                                    max_dis=18, min_dis=12)
    id_size = test_true_id_x.shape[0]
    i_index_set = np.arange(0, id_size, id_size / (i_max))[:i_max]

    inputs_ori_tensor = torch.stack([dset_test[test_true_id_x[i_index_set[i]]][0] for i in range(i_max)])

    from .myGetAdvVulnerable import get_unique_index
    j_index_matrix = get_unique_index(code, multi_label, j_max)

    adv_imgs = np.load(npy_path)
    ori_imgs = inputs_ori_tensor.cpu().numpy()
    imgs_test = np.load('save_for_load/imgs_test.npy')

    i=3
    j=7
    i_index = int(test_true_id_x[i_index_set[i]])
    j_index_set = j_index_matrix[int(test_true_label_y[i_index_set[i]])].astype(int)
    ori_img = ori_imgs[i]
    adv_img = adv_imgs[i, j]
    target_img = dset_database[j_index_set[j]][0].cpu().numpy()

    noise_level = 10
    perturbation_ratio = 1.0
    random_noise = np.random.randint(-noise_level, noise_level+1, ori_img.shape).astype(float)
    adv_img = ori_img + (adv_img - ori_img) * perturbation_ratio + random_noise/255
    #adv_img += random_noise / 255

    # center_img is used

    center_img = ori_img
    kNN_indexes = cal_knn_index_pixel(center_img, imgs_test, k=25)
    _, projected_img, kNN_imgs_mean = cal_geo_dist_github(ori_img, adv_img, imgs_test, k=25, return_projections=True, kNN_indexes=kNN_indexes)

    print("ori to center", np.linalg.norm((ori_img - center_img).reshape([-1]), ord=2))
    print("adv to center", np.linalg.norm((adv_img - center_img).reshape([-1]), ord=2))
    print("Projected to center:", np.linalg.norm((projected_img - center_img).reshape([-1]), ord=2))
    print("adv to ori:", np.linalg.norm((adv_img - ori_img).reshape([-1]), ord=2))
    print("Projected to adv:", np.linalg.norm((adv_img - projected_img).reshape([-1]), ord=2))
    print("Projected to ori:", np.linalg.norm((projected_img - ori_img).reshape([-1]), ord=2))

    if False:
        from .myRetrieval import get_img_num_by_class_from_img
        X = Variable(torch.Tensor(ori_img)).cuda().unsqueeze(0)
        img_num_result = get_img_num_by_class_from_img(X, model1, code, multi_label, threshold=5)
        print("Ori:", img_num_result.argmax() if img_num_result.sum()>0 else -1, img_num_result)
        X = Variable(torch.Tensor(adv_img)).cuda().unsqueeze(0)
        img_num_result = get_img_num_by_class_from_img(X, model1, code, multi_label, threshold=5)
        print("Adv:", img_num_result.argmax() if img_num_result.sum()>0 else -1, img_num_result)
        X = Variable(torch.Tensor(target_img)).cuda().unsqueeze(0)
        img_num_result = get_img_num_by_class_from_img(X, model1, code, multi_label, threshold=5)
        print("Target:", img_num_result.argmax() if img_num_result.sum()>0 else -1, img_num_result)
        X = Variable(torch.Tensor(projected_img)).cuda().unsqueeze(0)
        img_num_result = get_img_num_by_class_from_img(X, model1, code, multi_label, threshold=5)
        print("Projected:", img_num_result.argmax() if img_num_result.sum()>0 else -1, img_num_result)

        from scipy.spatial.distance import cosine
        print("Cosine between adv-target and adv-ori(less is better):", cosine((target_img-adv_img).reshape([-1]), (ori_img-adv_img).reshape([-1])))

        print("adv to ori", np.linalg.norm((ori_img - adv_img).reshape([-1]), ord=2))
        print("adv to target", np.linalg.norm((adv_img - target_img).reshape([-1]), ord=2))
        print("ori to target:", np.linalg.norm((ori_img - target_img).reshape([-1]), ord=2))

        plt.figure(666)
        plt.subplot(2,2,1)
        plt.imshow(np.moveaxis(ori_img, 0,-1))
        plt.subplot(2,2,2)
        plt.imshow(np.moveaxis(adv_img, 0,-1))
        plt.subplot(2,2,3)
        plt.imshow(np.moveaxis(projected_img, 0,-1))
        plt.subplot(2,2,4)
        plt.imshow(np.moveaxis(kNN_imgs_mean, 0, -1))

    return

if __name__ == "__main__":
    dis_method_value = ['cW', 'fW', 'cB', 'fB', 'cWcB', 'cWfB', 'fWcB', 'fWfB']
    dis_method = dis_method_value[0]
    i_max, j_max = 64, 32
    adv_method = 'miFGSMDI'
    step = 1.0
    linf = 32.0
    job_dataset = 'imagenet'
    net1 = 'ResNet152'
    net2 = 'ResNext101_32x4d'
    noiid = False

    main_func()