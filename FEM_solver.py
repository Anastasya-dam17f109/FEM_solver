import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import os
import os.path
import json

# набор узлов и получающихся КЭ
nodes = {}
mesh_elems = {}
# форма области задана жестко - это прямоугльник, строящийся по двум точкам coord_min и coord_max
coord_min = np.array([0,0], float)
coord_max = np.array([200, 100], float)

# служебная информация, диспользуется непосредственно в решении
mesh_B_dict ={}
mesh_list_dict = {}
teta_grad = {}
teta = []
q = {}

with open("task.json", 'r') as task_file:
	task = json.load(task_file)

# в качестве КЭ используется равнобедренный прямоугольный треугольник, длинна стороны которго = h
h = task["h"]

# параметры  области
lambda_ = task["lambda"]
alpha_t = task["alpha_t"]
f_h = task["f_h"]
q_h = task["q_h"]
teta_inf = task["teta_inf"]
'''
    учет  граничных условий - считаем, что возможны только условия 2го и 3го рода
    При этом помним, что ГУ 2го рода - это то же самое, что и ГУ 3го с 0-вой конвективной частью
    справа заданы ГУ 3го рода
    q = q_h;  alpha = alpha_t
    слева заданы ГУ 2го рода:
    q = q_h;  alpha = 0
    сверху и снизу заданы ГУ 2го рода:
    q =  0; alpha = 0

    получается, в задаче рассматриваются как бы 3 поверхности, на которых разные значения q и  alpha
    упорядочим эти поверхности - введем для каждой индекс:
     0 - правая граница
     1 - левая граница
     2 - верхняя или нижняя граница

     значения q и  alpha для каждой границы запишутся в массивы
     по соответствующим индексам.
     эти массивы используются непосредственно при решении задачи.

'''
q_bounds = [q_h, q_h, 0]
alpha_bounds = [0, alpha_t, 0]

# триангуляция заданной области

def triangulation():
    #создание узлов
    coord_buf = np.zeros(2, dtype=float)
    coord_up = np.zeros(2, dtype=int)
    coord_down = np.zeros(2, dtype=int)
    x_n = int((coord_max[0]-coord_min[0])// h)+1
    if (coord_max[0]-coord_min[0]) / h >x_n:
        x_n += 1
    y_n = int((coord_max[1]-coord_min[1]) // h)+1
    if (coord_max[1]-coord_min[1]) / h > x_n:
        y_n += 1
    counter = 0
    coord_buf[1] = coord_min[1]
    for i in range(x_n):
        coord_buf[1] = coord_max[1]
        for j in range(y_n):
            nodes.update({counter : coord_buf.copy()})
            counter += 1
            if j != y_n-1:
                coord_buf[1] -= h
            else:
                coord_buf[1] = coord_min[1]
        if i != x_n - 1:
            coord_buf[0] += h
        else:
            coord_buf[0] = coord_max[0]
    counter_mesh = 0
    # создание конечных элементов
    for i in range(x_n-1):
        for k in range(y_n-1):
            for l in range(2):
                coord_up[l] = (i + l)*y_n+k
                coord_down[l] = (i + l)*y_n + k + 1
            mesh_elems.update({counter_mesh    : np.array([coord_up[0],coord_up[1],coord_down[0]])})
            mesh_elems.update({counter_mesh + 1: np.array([coord_up[1], coord_down[0], coord_down[1]])})
            counter_mesh += 2

# функция , осуществляющая проверку того, что узлы выбранного КЭ лежат на выбранной границе
# непосредственное осуществление проверки в соответствии с заданными признаками

def on_sigma_nodes_list(treshold, idx_el,idx_coord):
    node_list = mesh_elems.get(idx_el)
    curve = []
    buf = 0
    for i in range(3):
        if nodes.get(node_list[i])[idx_coord] == treshold:
            curve.append(node_list[i])
        else:
            buf = node_list[i]
    if len(curve) == 2:
        curve.append(buf)
        return curve
    else:
        return []

# функция , осуществляющая проверку того, что узлы выбранного КЭ лежат на выбранной границе -
# в данной функции задается набор признаков для проверки

def on_sigma_nodes(idx_surf, idx_el):
    # список узлов конечного элемента, лежащих на левой границе
    if idx_surf == 0:
        return on_sigma_nodes_list(coord_min[0], idx_el, 0)
    else:
        # список узлов конечного элемента, лежащих на правой границе
        if idx_surf == 1:
            return on_sigma_nodes_list(coord_max[0], idx_el, 0)
        else:
            # список узлов конечного элемента, лежащих на верхней и нижней границе
            buf_curve = on_sigma_nodes_list(coord_min[1], idx_el, 1)
            if len(buf_curve) != 0:
                return buf_curve
            else:
                return on_sigma_nodes_list(coord_max[1], idx_el, 1)



# вычисление барицентрических координат  узлов элемента

def baricentric(node_list):
    v_matrix = np.ones(9, dtype = float).reshape(3,3)
    for i in range(3):
        v_matrix[i,1:] = nodes.get(node_list[i])
    return np.linalg.inv(v_matrix.T)


# вычисление площади конечного элемента

def calc_S(v1,v2,v3):
    p1 = np.array(nodes[v1])
    p2 = np.array(nodes[v2])-p1
    p3 = np.array(nodes[v3])-p1
    v_res = np.cross(p2,p3)
    return np.sqrt(np.dot(v_res,v_res))*0.5

# вычисление длины стороны конечного элемента

def calc_L(v1,v2):
    v_res = np.array(nodes[v1]) - np.array(nodes[v2])
    return np.sqrt(np.dot(v_res,v_res))

# сборка глобальной матрицы

def create_global_system():
    mesh_L_dict = {}
    mesh_zero_coord_dict = {}
    mesh_S_dict = {}
    M_buf = np.array(
        [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0, ], [1.0, 1.0, 2.0, ]])
    for i in range(len(mesh_elems)):
        node_list = mesh_elems.get(i).copy()
        S = calc_S(node_list[0], node_list[1], node_list[2])
        mesh_S_dict.update({i: S})
        f = (f_h * S / 3) * np.array([1.0, 1.0, 1.0])
        L = [0, 0, 0]
        zero_coord = [0, 0, 0]
        node_list = []
        B_flag = False

        for j in range(3):
            sigma_s_elems = on_sigma_nodes(j, i)
            if len(sigma_s_elems) != 0:
                L[j] = calc_L(sigma_s_elems[0], sigma_s_elems[1])
                if not B_flag :
                    B_flag =True
                    node_list.append(sigma_s_elems[2])
                    node_list.append(sigma_s_elems[0])
                    node_list.append(sigma_s_elems[1])
                    B = baricentric(node_list.copy())
                    mesh_B_dict.update({i: B})
                    mesh_list_dict.update({i: node_list})
                else:
                    zero_coord[j] =  node_list.index(sigma_s_elems[2])
            else:
                if j ==2:
                    if not B_flag:
                        B_flag = True
                        node_list = mesh_elems.get(i).copy()
                        B = baricentric(node_list.copy())
                        mesh_B_dict.update({i: B})
                        mesh_list_dict.update({i: node_list})

            buf_vec = np.array([1.0,1.0,1.0])
            buf_vec[zero_coord[j]] = 0;
            f -= ((q_bounds[j]+alpha_bounds[j]*teta_inf)*L[j]/2)*buf_vec
        mesh_L_dict.update({i: L.copy()})
        mesh_zero_coord_dict.update({i: zero_coord})
        for j in range(3):
            f_gl_std[node_list[j]] += f[j]

    for i in range(len(mesh_elems)):
        B = mesh_B_dict.get(i).copy().T
        B_t = mesh_B_dict.get(i)[:, 1:]
        B = B[1:, :]
        M_list = [M_buf.copy(),M_buf.copy(),M_buf.copy() ]
        for j in range(3):
            zero_idx = mesh_zero_coord_dict.get(i)[j]
            for k in range(3):
                M_list[j][zero_idx][k] = 0
                M_list[j][k][zero_idx] = 0

        G = lambda_ * mesh_S_dict.get(i) * np.dot(B_t, B) - alpha_bounds[0] * (mesh_L_dict.get(i)[0] / 6) * M_list[0] \
            - alpha_bounds[1] * (mesh_L_dict.get(i)[1] / 6) * M_list[1]- alpha_bounds[2] * (mesh_L_dict.get(i)[2] / 6) * M_list[2]

        for j in range(3):
            i_gl = mesh_list_dict.get(i)[j]
            for k in range(3):
                j_gl = mesh_list_dict.get(i)[k]
                gl_matr_std[i_gl, j_gl] += G[j, k]


# вычисление аппроксимации q и grad_teta

def find_approximation():
    for i in range(len(mesh_list_dict)):
        node_list = mesh_list_dict.get(i)
        teta_list = np.array([[0.0],[0.0],[0.0]])
        for j in range(3):
            teta_list[j,0] = teta[node_list[j]]
        buf_teta = np.dot(mesh_B_dict.get(i).T[1:, :], teta_list)
        buf_teta = [buf_teta[0,0],buf_teta[1,0]]
        for j in range(3):
            if teta_grad.get(node_list[j])is not None:
                p = teta_grad.get(node_list[j])
                p.append(buf_teta)
                teta_grad.update({node_list[j]: p})
            else:
                teta_grad.update({node_list[j]: [buf_teta]})

    for i in range(len(nodes)):
         length = len(teta_grad.get(i))
         buf_teta1 = np.zeros(2)
         for j in range(length):
             for k in range(2):
                buf_teta1[k] += teta_grad.get(i)[j][k]/length
         teta_grad.update({i: buf_teta1.copy()})
         q.update({i: buf_teta1.copy()*lambda_})

# вывод результата в файл формата mv2 - для визуаизации

def print_in_mv2():
    with open("./result_mke.txt",'w') as file:
        file.write(str(len(nodes))+' 3 5 teta grad_t_x grad_t_y  q_x q_y  \n')
        for i in range(len(nodes)):
            str_buf1 = ''
            str_buf3 = ''
            str_buf4 = ''
            for j in range(2):
                str_buf1 += str(nodes.get(i)[j])+' '
                str_buf3 += str(teta_grad.get(i)[j]) + ' '
                str_buf4 += str(q.get(i)[j]) + ' '
            str_buf1 += '0 '
            file.write( str(i+1) + ' ' + str_buf1 + str(teta[i]) + ' ' + str_buf3 + str_buf4 + '\n')
        file.write(str(len(mesh_elems)) + ' 3 3 BC_id mat_id mat_id_Out\n')
        for i in range(len(mesh_elems)):
            str_buf1 = ''
            for j in range(3):
                str_buf1 += str(mesh_list_dict.get(i)[j] + 1) + ' '
            file.write(str(i + 1) + ' ' + str_buf1 + '0 1 0\n')

    file.close()
    if os.path.exists("./result_mke.mv2"):
        os.remove("./result_mke.mv2")
    os.rename("./result_mke.txt","./result_mke.mv2")


# решение поставленной задачи

triangulation()
f_gl_std = np.zeros(len(nodes))
gl_matr_std   = np.zeros(len(nodes)*len(nodes)).reshape(len(nodes),len(nodes))
create_global_system()
teta = np.linalg.solve(gl_matr_std , f_gl_std)
find_approximation()
print_in_mv2()
