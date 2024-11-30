#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

fstream dbg;

constexpr int INITIAL_DISTR_TAG_META_OR_FINISHED = 1;
constexpr int INITIAL_DISTR_TAG_V = 2;
constexpr int INITIAL_DISTR_TAG_C = 3;
constexpr int SYNC_AFTER_INITIAL_DISTR = 4;
constexpr int PRINT_ALL_ACCUMS = 5;
constexpr int FAKE_ALL_TO_ALL = 6;

constexpr bool MUL_DBG = 0;
constexpr bool PLUSEQ_DBG = 0;
constexpr bool MUL_RESULT_DBG = 0;
constexpr bool CALC_LOCAL_DIMS_DBG = 0;
constexpr bool SUMMA_2D_DBG = 0;
constexpr bool MERGE_ACCUMS_DBG = 0;

int numProcesses = 0;
int myRank = 0;
int myLayer = 0;
int layerOrder = 0;
int my2d = 0;
int proc_ct_sqrt = 1;

struct CSRM {
    vector<double> v;
    vector<int> c, r;
    int rows, cols, nnz, mnzr;

    CSRM() {
        set_matrix_dims(vector<int>(4, 0).data());
        r.push_back(0);
    }

    void set_rowscols(int *data) {
        rows = data[0];
        cols = data[1];
    }

    int ct_larger(double g) {
        int result = 0;
        for (auto x : v)
            result += x > g;
        return result;
    }

    void set_matrix_dims(int *data) {
        rows = data[0]; cols = data[1]; nnz = data[2]; mnzr = data[3];
    }

    void append_row(int len, vector<double> &&vv, vector<int> &&cc, int row_nr) {
        int missing_rows = row_nr - r.size() + 1;
        if (missing_rows > 0) {
            auto tmp = vector<int>(missing_rows, r.back());
            r.insert(r.end(), tmp.begin(), tmp.end());
        }
        v.insert(v.end(), vv.begin(), vv.end());
        c.insert(c.end(), cc.begin(), cc.end());
        r.push_back(r.back() + len);
    }

    void to_rsc() {
        vector<double> vv;
        vector<int> cc, rr;
        int ct = 0;
        rr.push_back(0);

        for (int i = 0; i < cols; i++) {
            int row = 0;

            for (int j = 0; j < (int)c.size(); j++) {
                while (row != (int)r.size() - 1 && j >= r[row + 1])
                    row++;

                if (c[j] == i) {
                    ++ct;
                    cc.push_back(row);
                    vv.push_back(v[j]);
                }
            }
            rr.push_back(ct);
        }
        
        swap(r, rr);
        swap(v, vv);
        swap(c, cc);
        swap(rows, cols);
    }

    CSRM operator *(CSRM &other) {
        CSRM x;
        int total = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < other.rows; j++) {
                double dtmp = 0;
                int as = r[i];
                int bs = other.r[j];
                int ae = r[i + 1];
                int be = other.r[j + 1];
                if (MUL_DBG) dbg << i << " " << as << " " << ae << " vs " << j << " " << bs << " " << be << endl;
                for (int aptr = as, bptr = bs; aptr < ae && bptr < be;) {
                    if (MUL_DBG) dbg << i << " " << j << "   " << c[aptr] << " " << other.c[bptr] << "   " << v[aptr] << " " << other.v[bptr] << endl; 
                    if (c[aptr] < other.c[bptr]) {
                        ++aptr;
                        continue;
                    }
                    if (other.c[bptr] < c[aptr]) {
                        ++bptr;
                        continue;
                    }
                    dtmp += v[aptr] * other.v[bptr];
                    ++aptr;
                    ++bptr;
                }
                if (dtmp) {
                    ++total;
                    x.v.push_back(dtmp);
                    x.c.push_back(j);
                }
            }
            x.r.push_back(total);
        }
        return x;
    }

    CSRM& operator +=(CSRM &&other) {
        CSRM tmp;

        auto append_last = [&](vector<double> &appended_v, vector<int> &appended_c, int &ptr, int &appended_len) {
            tmp.v.insert(tmp.v.end(), appended_v.begin() + ptr, appended_v.begin() + ptr + appended_len);
            tmp.c.insert(tmp.c.end(), appended_c.begin() + ptr, appended_c.begin() + ptr + appended_len);
            ptr += appended_len;
            tmp.r[tmp.r.size() - 1] += appended_len;
            appended_len = 0;
        };
        if (PLUSEQ_DBG) dbg << "plus " << endl;
        int local_ptr = 0, global_ptr = 0;
        for (int row = 0; row < (int)r.size() - 1; row++) {
            int loc_row_len = other.r[row + 1] - other.r[row], glob_row_len = r[row + 1] - r[row];
            if (PLUSEQ_DBG) dbg << row << " " << loc_row_len << " " << glob_row_len << endl;
            tmp.r.push_back(tmp.r.back());
            for (; loc_row_len || glob_row_len; tmp.r[tmp.r.size() - 1]++) {
                if (!loc_row_len) {
                    append_last(v, c, global_ptr, glob_row_len);
                    break;
                }
                if (!glob_row_len) {
                    append_last(other.v, other.c, local_ptr, loc_row_len);
                    break;
                }   
                if (PLUSEQ_DBG) dbg << row << "   " << other.c[local_ptr] << " " << other.v[local_ptr] << "   " << c[global_ptr] << " " << v[global_ptr] << endl;
                if (other.c[local_ptr] == c[global_ptr]) {
                    double sum = other.v[local_ptr] + v[global_ptr];
                    if (sum < 0 || sum > 0) {
                        tmp.v.push_back(other.v[local_ptr] + v[global_ptr]);
                        tmp.c.push_back(other.c[local_ptr]);
                    } else {
                        --tmp.r[tmp.r.size() - 1];
                    }
                    --loc_row_len;
                    --glob_row_len;
                    ++local_ptr;
                    ++global_ptr;
                    continue;
                }
                if (other.c[local_ptr] < c[global_ptr]) {
                    tmp.v.push_back(other.v[local_ptr]);
                    tmp.c.push_back(other.c[local_ptr]);
                    --loc_row_len;
                    ++local_ptr;
                    continue;
                }
                tmp.v.push_back(v[global_ptr]);
                tmp.c.push_back(c[global_ptr]);
                --glob_row_len;
                ++global_ptr;
            }
        }
        if (PLUSEQ_DBG) dbg << "plus end" << endl << endl;
        swap(tmp.v, v);
        swap(tmp.c, c);
        swap(tmp.r, r);
        return *this;
    }
};

CSRM A, B; // just dims
CSRM a, b, accum; // local matrices

int myRow, myCol;

template<typename T>
void debug_vect(vector<T> &x) {
    for (auto &xx : x) 
        dbg << xx << " ";
    dbg << endl;
}

inline MPI_Status get_new_status() {
    MPI_Status s;
    s.MPI_ERROR = 0;
    return s;
}

inline void check_mpi_status(MPI_Status &status, int line) {
    if (status.MPI_ERROR != MPI_SUCCESS) {
        fprintf(stderr, "MPI error %d when receiving from %d, tag: %d, occurred at line: %d\n", 
            status.MPI_ERROR, status.MPI_SOURCE, status.MPI_TAG, line);
    }
}

void set_matrix_dims(int *data, bool is_a) {
    if (is_a) {
        A.set_matrix_dims(data);
    } else {
        B.set_matrix_dims(data);
    }
}

void send_matrix_dims(int rows, int cols, int nnz, int mnzr, bool is_a) {
    int data[4] = {rows, cols, nnz, mnzr};
    MPI_Bcast(data, 4, MPI_INT, 0, MPI_COMM_WORLD);
    set_matrix_dims(data, is_a);
}

void receive_matrix_dims(bool is_a) {
    int data[4];
    MPI_Bcast(data, 4, MPI_INT, 0, MPI_COMM_WORLD);
    set_matrix_dims(data, is_a);
}

bool receive_matrix(bool is_a) {
    int meta[2] = {0, 0};
    auto status = get_new_status();
    MPI_Request requests[3];
    MPI_Status statuses[3] = {get_new_status(), get_new_status(), get_new_status()};
    MPI_Request meta_request;

    MPI_Irecv(meta, 2, MPI_INT, 0, INITIAL_DISTR_TAG_META_OR_FINISHED, MPI_COMM_WORLD, &meta_request);
    MPI_Wait(&meta_request, &status);
    check_mpi_status(status, __LINE__);
    // dbg << meta[0] << " received in keep receiving " << meta[1] << " is_a: " << is_a << endl;

    (is_a ? a : b).v = vector<double>(meta[0], 0);
    (is_a ? a : b).c = vector<int>(meta[0], 0);
    (is_a ? a : b).r = vector<int>(meta[1], 0);
    
    MPI_Irecv((is_a ? a : b).v.data(), meta[0], MPI_DOUBLE, 0, INITIAL_DISTR_TAG_V, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv((is_a ? a : b).c.data(), meta[0], MPI_INT, 0, INITIAL_DISTR_TAG_C, MPI_COMM_WORLD, &requests[1]);
    MPI_Irecv((is_a ? a : b).r.data(), meta[1], MPI_INT, 0, INITIAL_DISTR_TAG_C, MPI_COMM_WORLD, &requests[2]);
    MPI_Waitall(3, requests, statuses);
    return true;
}

// naive functions dont take weird B ordering into account.
long long get_naive_proc_rank_for_tight(int col_nr, int dim, int layers) {
    return ((long long)proc_ct_sqrt * layers * col_nr) / dim;
}

long long get_naive_proc_rank_for_wide(int row_nr, int dim) {
    return ((long long)proc_ct_sqrt * row_nr) / dim;
}

void read(string path, int proc_ct_sqrt, int layers, bool is_a) {
    int rows, cols, nnz, mnzr;
    int dim; // cols == rows so we take either one.

    auto get_proc_for_tight = [&proc_ct_sqrt, &dim, &layers](int col_nr) {
        return get_naive_proc_rank_for_tight(col_nr, dim, layers);
    };

    auto get_proc_for_wide = [&proc_ct_sqrt, &dim](int row_nr) {
        return get_naive_proc_rank_for_wide(row_nr, dim);
    };

    auto get_column_in_proc_grid = [&is_a, &get_proc_for_tight, &get_proc_for_wide](int col){
        if (is_a)
            return get_proc_for_tight(col);
        else    
            return get_proc_for_wide(col);
    };

    auto get_row_in_proc_grid = [&is_a, &get_proc_for_tight, &get_proc_for_wide](int row){
        if (is_a)
            return get_proc_for_wide(row);
        else    
            return get_proc_for_tight(row);
    };

    auto get_proc = [&proc_ct_sqrt, &dim, &is_a, &layers](int row_nr, int col_proc) {
        if (is_a) 
            return ((long long)proc_ct_sqrt * row_nr) / dim * layers * proc_ct_sqrt + col_proc;  
        long long layered = ((long long) proc_ct_sqrt * layers * row_nr) / dim; // layered % layers gives layer in a proc square
        long long res = (layered % layers) + col_proc * layers + ((layered / layers) * layers * proc_ct_sqrt);  
        // dbg << row_nr << " kasia " << col_proc << " " << res << " layered: " << layered << endl;
        return res;
    };

    ifstream matrix_file(path);
    if (!matrix_file.is_open()) {
        dbg << "matrix file " << path << " cannot be opened";
        exit(1);
    }

    matrix_file >> rows >> cols >> nnz >> mnzr;
    dim = rows;
    send_matrix_dims(rows, cols, nnz, mnzr, is_a);
    vector<double> v(nnz);
    vector<int> c(nnz), r(rows + 1);
    for (int i = 0; i < nnz; i++)
        matrix_file >> v[i];
    for (int i = 0; i < nnz; i++)
        matrix_file >> c[i];
    for (int i = 0; i < rows + 1; i++)
        matrix_file >> r[i];
    
    CSRM split[numProcesses];

    auto get_proc_for_col = [&](long long col) {
        return (layers * col) / cols;
    };

    // copied from calc_local_dims!!! FIXME
    auto get_first = [&](long long id) {
        long long p = 0, k = cols - 1, mid;
        while (p <= k) {
            mid = (p + k) >> 1;
            long long x = get_proc_for_col(mid);
            if (x < id) {
                ++p;
                mid = p;
            } else {
                mid = k;
                k--;
            }
        }
        return mid;
    };
    
    for (int row_nr = 0; row_nr < rows; row_nr++) {
        if (is_a) {
            int fproc = get_proc(row_nr, 0);
            for (int i = fproc; i < fproc + layers * proc_ct_sqrt; i++)
                split[i].r.push_back(split[i].r.back());
        } else {
            int fproc = get_proc(row_nr, 0);
            for (int i = fproc; i < fproc + layers * proc_ct_sqrt; i += layers)
                split[i].r.push_back(split[i].r.back());
        }

        for (int i = r[row_nr]; i < r[row_nr + 1]; i++) {
            long long id = get_proc(row_nr, get_column_in_proc_grid(c[i]));
            split[id].v.push_back(v[i]);
            split[id].c.push_back(c[i]);
            split[id].r.back()++;
        }
    }

    (is_a ? a : b).v = split[0].v;
    (is_a ? a : b).c = split[0].c;
    (is_a ? a : b).r = split[0].r;
    
    MPI_Request requests[numProcesses * 3];
    MPI_Status statuses[numProcesses * 3];

    MPI_Request data_requests[numProcesses];
    MPI_Status data_statuses[numProcesses];
    int dims[numProcesses][2];
    for (int i = 1; i < numProcesses; i++) {
        dims[i][0] = split[i].c.size();
        dims[i][1] = split[i].r.size();
        data_statuses[i] = get_new_status();
        for (int j = 0; j < 3; j++)
            statuses[i * 3 + j] = get_new_status();
    }

    for (int i = 1; i < numProcesses; i++) 
        MPI_Isend(dims[i], 2, MPI_INT, i, INITIAL_DISTR_TAG_META_OR_FINISHED, MPI_COMM_WORLD, &data_requests[i]);
    MPI_Waitall(numProcesses - 1, data_requests + 1, data_statuses + 1);

    for (int i = 1; i < numProcesses; i++) {
        MPI_Isend(split[i].v.data(), split[i].v.size(), MPI_DOUBLE, i, INITIAL_DISTR_TAG_V, MPI_COMM_WORLD, &requests[i * 3]);
        MPI_Isend(split[i].c.data(), split[i].c.size(), MPI_INT, i, INITIAL_DISTR_TAG_C, MPI_COMM_WORLD, &requests[i * 3 + 1]);
        MPI_Isend(split[i].r.data(), split[i].r.size(), MPI_INT, i, INITIAL_DISTR_TAG_C, MPI_COMM_WORLD, &requests[i * 3 + 2]);
    }
    MPI_Waitall(numProcesses * 3 - 3, requests + 3, data_statuses + 3);
}

void calc_local_dims() {
    // TODO calc mnzr? nnz?
    auto find_proc_rowcol = [&](int rank, int big_matrix_rowcol) {
        return (rank * big_matrix_rowcol + proc_ct_sqrt - 1) / proc_ct_sqrt;
    };
    
    a.rows = find_proc_rowcol(myRank / proc_ct_sqrt + 1, A.rows) - find_proc_rowcol(myRank / proc_ct_sqrt, A.rows);
    a.cols = find_proc_rowcol(myRank % proc_ct_sqrt + 1, A.cols) - find_proc_rowcol(myRank % proc_ct_sqrt, A.cols);
    // from assumption Arows = Acols = Bcols = Brows:
    b.rows = a.rows;
    b.cols = a.cols;

    if (CALC_LOCAL_DIMS_DBG) dbg << find_proc_rowcol(myRank / proc_ct_sqrt, A.rows) << " " << find_proc_rowcol(myRank % proc_ct_sqrt, A.cols) << " " << 
        find_proc_rowcol(myRank / proc_ct_sqrt + 1, A.rows) << " " << find_proc_rowcol(myRank % proc_ct_sqrt + 1, A.cols) << endl;
    
    int ca0 = find_proc_rowcol(myRank % proc_ct_sqrt, A.cols);
    if (CALC_LOCAL_DIMS_DBG) dbg << a.cols << " shift " << ca0 << endl;
    for (auto &c : a.c) 
        c = (c + a.cols - ca0) % a.cols;

    // its before the transposition
    int cb0 = find_proc_rowcol(myRank % proc_ct_sqrt, A.cols);
    if (CALC_LOCAL_DIMS_DBG) dbg << b.cols << " shift " << cb0 << endl;
    debug_vect(b.c);
    for (auto &c : b.c)
        c = (c + b.cols - cb0) % b.cols;
    debug_vect(b.c);
    
    if (CALC_LOCAL_DIMS_DBG) dbg << a.rows << " " << b.rows << "   " << a.r.size() << " " << b.r.size() << endl;
    while ((int)a.r.size() < a.rows + 1) 
        a.r.push_back(a.r.back());
    while ((int)b.r.size() < b.rows + 1) 
        b.r.push_back(b.r.back());
}

void calc_local_dims(int layers) {
    // TODO calc mnzr? nnz?

    /*
    A:
    0 0 1 2 2 3
    0 0 1 2 2 3
    0 0 1 2 2 3
    4 4 5 6 6 7
    4 4 5 6 6 7
    4 4 5 6 6 7

    B:
    0 0 0 2 2 2 
    0 0 0 2 2 2 
    1 1 1 3 3 3 
    4 4 4 6 6 6
    4 4 4 6 6 6
    5 5 5 7 7 7
    */
    int proc_grid_col = layerOrder % proc_ct_sqrt, proc_grid_row = layerOrder / proc_ct_sqrt;
    // (col, row, myLayer)
    int big_dim = A.cols; // A.cols = A.rows = B.cols = B.rows;

    auto get_first_tight = [&](int id) {
        int p = 0, k = big_dim - 1, mid;
        while (p <= k) {
            mid = (p + k) >> 1;
            auto x = get_naive_proc_rank_for_tight(mid, big_dim, layers);
            if (x < id) {
                ++p;
                mid = p;
            } else {
                mid = k;
                k--;
            }
        }
        return mid;
    };

    auto get_first_wide = [&](int id) {
        int p = 0, k = big_dim - 1, mid;
        while (p <= k) {
            mid = (p + k) >> 1;
            auto x = get_naive_proc_rank_for_wide(mid, big_dim);
            if (x < id) {
                ++p;
                mid = p;
            } else {
                mid = k;
                k--;
            }
        }
        return mid;
    };

    int acol_id = proc_grid_col * layers + myLayer;
    int brow_id = proc_grid_row * layers + myLayer;
    a.rows = get_first_wide(proc_grid_row + 1) - get_first_wide(proc_grid_row);
    a.cols = get_first_tight(acol_id + 1) - get_first_tight(acol_id);
    b.rows = get_first_tight(brow_id + 1) - get_first_tight(brow_id);
    b.cols = get_first_wide(proc_grid_col + 1) - get_first_wide(proc_grid_col);

    if (CALC_LOCAL_DIMS_DBG) {
        dbg << "a rows: " << get_first_wide(proc_grid_row) << " " << get_first_wide(proc_grid_row + 1) << endl;
        dbg << "a cols: " << get_first_tight(acol_id) << " " << get_first_tight(acol_id + 1) << endl;
        dbg << "b rows: " << get_first_tight(brow_id) << " " << get_first_tight(brow_id + 1) << endl;
        dbg << "b cols: " << get_first_wide(proc_grid_col) << " " << get_first_wide(proc_grid_col + 1) << endl;
    }

    int ca0 = get_first_tight(acol_id);
    if (CALC_LOCAL_DIMS_DBG) dbg << a.cols << " shift " << ca0 << endl;
    for (auto &c : a.c) 
        c = (c + a.cols - ca0) % a.cols;

    // its before the transposition
    int cb0 = get_first_wide(proc_grid_col);
    if (CALC_LOCAL_DIMS_DBG) dbg << b.cols << " shift " << cb0 << endl;
    if (CALC_LOCAL_DIMS_DBG) debug_vect(b.c);
    for (auto &c : b.c)
        c = (c + b.cols - cb0) % b.cols;
    if (CALC_LOCAL_DIMS_DBG) debug_vect(b.c);
    
    if (CALC_LOCAL_DIMS_DBG) dbg << a.rows << " " << b.rows << "   " << a.cols << " " << b.cols << endl;
    while ((int)a.r.size() < a.rows + 1) 
        a.r.push_back(a.r.back());
    while ((int)b.r.size() < b.rows + 1) 
        b.r.push_back(b.r.back());
}

void mul(CSRM &row_bc, CSRM &col_bc) {
    accum += row_bc * col_bc;
    
    if (MUL_RESULT_DBG) {
        dbg << "step done" << endl;
        debug_vect(accum.v);
        debug_vect(accum.c);
        debug_vect(accum.r);
    }
}

void summa_2d(MPI_Comm comm, int layers = 1) {
    int summaRank = myRank / layers;
    myRow = summaRank / proc_ct_sqrt;
    myCol = summaRank % proc_ct_sqrt;
    
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(comm, myRow, myCol, &row_comm);
    MPI_Comm_split(comm, myCol, myRow, &col_comm);
    // initialize accumulatos
    accum.r = vector<int>(a.rows + 1);
    accum.set_rowscols(vector<int>({a.rows, b.rows}).data());

    for (int round = 0; round < proc_ct_sqrt; round++) {
        CSRM row_bc, col_bc;
        int data_sent_in_row[6], data_sent_in_col[6];

        if (myCol == round) {
            if (SUMMA_2D_DBG) dbg << "gonna send row " << round << " " << summaRank << endl;
            row_bc.v = vector<double>(a.v);
            row_bc.c = vector<int>(a.c);
            row_bc.r = vector<int>(a.r);
            
            if (SUMMA_2D_DBG) {
                debug_vect(row_bc.v);
                debug_vect(row_bc.c);
                debug_vect(row_bc.r);
            }

            data_sent_in_row[0] = row_bc.v.size();
            data_sent_in_row[1] = row_bc.c.size();
            data_sent_in_row[2] = row_bc.r.size();
            data_sent_in_row[3] = a.rows;
            data_sent_in_row[4] = a.cols;
            data_sent_in_row[5] = myRank;
        }
        MPI_Bcast(data_sent_in_row, 6, MPI_INT, round, row_comm);
        if (myCol != round) {
            row_bc.v = vector<double>(data_sent_in_row[0]);
            row_bc.c = vector<int>(data_sent_in_row[1]);
            row_bc.r = vector<int>(data_sent_in_row[2]);
        }
        row_bc.set_rowscols(data_sent_in_row + 3);

        // FIXME scalić komunikaty / porównać czasy działania
        MPI_Bcast(row_bc.v.data(), data_sent_in_row[0], MPI_DOUBLE, round, row_comm);
        MPI_Bcast(row_bc.c.data(), data_sent_in_row[1], MPI_INT, round, row_comm);
        MPI_Bcast(row_bc.r.data(), data_sent_in_row[2], MPI_INT, round, row_comm);

        if (myRow == round) {
            col_bc.v = vector<double>(b.v);
            col_bc.c = vector<int>(b.c);
            col_bc.r = vector<int>(b.r);
            if (SUMMA_2D_DBG) {
                dbg << "gonna send col " << round << " " << summaRank << endl;
                debug_vect(col_bc.v);
                debug_vect(col_bc.c);
                debug_vect(col_bc.r);
            }
            data_sent_in_col[0] = col_bc.v.size();
            data_sent_in_col[1] = col_bc.c.size();
            data_sent_in_col[2] = col_bc.r.size();
            data_sent_in_col[3] = b.rows;
            data_sent_in_col[4] = b.cols;
            data_sent_in_col[5] = myRank;
        }
        MPI_Bcast(data_sent_in_col, 6, MPI_INT, round, col_comm);
        if (myRow != round) {
            col_bc.v = vector<double>(data_sent_in_col[0]);
            col_bc.c = vector<int>(data_sent_in_col[1]);
            col_bc.r = vector<int>(data_sent_in_col[2]);
        }
        col_bc.set_rowscols(data_sent_in_col + 3);
        // FIXME scalić komunikaty / porównać czasy działania
        MPI_Bcast(col_bc.v.data(), data_sent_in_col[0], MPI_DOUBLE, round, col_comm);
        MPI_Bcast(col_bc.c.data(), data_sent_in_col[1], MPI_INT, round, col_comm);
        MPI_Bcast(col_bc.r.data(), data_sent_in_col[2], MPI_INT, round, col_comm);

        if (SUMMA_2D_DBG) {
            dbg << "A sent: " << round << " " << summaRank << " from " << data_sent_in_row[5] << endl; 
            dbg << row_bc.rows << " " << row_bc.cols << " " << data_sent_in_row[0] << " " << data_sent_in_row[1] << " " << data_sent_in_row[2] << endl;
            debug_vect(row_bc.v);
            debug_vect(row_bc.c);
            debug_vect(row_bc.r);
            dbg << endl;
            dbg << "B sent: " << round << " " << summaRank << " from " << data_sent_in_col[5] << endl;
            dbg << col_bc.rows << " " << col_bc.cols << " " << data_sent_in_col[0] << " " << data_sent_in_col[1] << " " << data_sent_in_col[2] << endl;
            debug_vect(col_bc.v);
            debug_vect(col_bc.c);
            debug_vect(col_bc.r);
            dbg << endl;
        }
        if (data_sent_in_row[4] != data_sent_in_col[4]) {
            dbg << "mismatching dimensions in summa2d - case a: " << data_sent_in_row[3] << "x" << data_sent_in_row[4] << " vs "
                << data_sent_in_col[3] << "x" << data_sent_in_col[4] << endl; 
            exit(1);
        }
        if (data_sent_in_row[3] != a.rows) {
            dbg << "mismatching dimensions in summa2d - case b: " << data_sent_in_row[3] << "x" << data_sent_in_row[4] << " vs "
                << a.rows << "x" << a.cols << endl; 
            exit(1);
        }
        if (data_sent_in_col[3] != b.rows) {
            dbg << "mismatching dimensions in summa2d - case c: " << data_sent_in_col[3] << "x" << data_sent_in_col[4] << " vs "
                << b.rows << "x" << b.cols << endl; 
            exit(1);
        }

        mul(row_bc, col_bc); 
    }
    
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
}

void print_all(int layers) {
    if (myRank) {
        int data[4] = {accum.rows, accum.cols, (int)accum.v.size(), (int)accum.r.size()};
        MPI_Send(data, 4, MPI_INT, 0, PRINT_ALL_ACCUMS, MPI_COMM_WORLD);
        MPI_Send(accum.v.data(), (int)accum.v.size(), MPI_DOUBLE, 0, PRINT_ALL_ACCUMS, MPI_COMM_WORLD);
        MPI_Send(accum.c.data(), (int)accum.c.size(), MPI_INT, 0, PRINT_ALL_ACCUMS, MPI_COMM_WORLD);
        MPI_Send(accum.r.data(), (int)accum.r.size(), MPI_INT, 0, PRINT_ALL_ACCUMS, MPI_COMM_WORLD);
        return;
    } 
    CSRM big[numProcesses];
    big[0] = accum;
    // the matrix is small, we can be lazy.
    for (int i = 1; i < numProcesses; i++) {
        int data[4];
        MPI_Recv(data, 4, MPI_INT, i, PRINT_ALL_ACCUMS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        big[i].v = vector<double>(data[2]);
        big[i].c = vector<int>(data[2]);
        big[i].r = vector<int>(data[3]);
        big[i].set_rowscols(data);
        MPI_Recv(big[i].v.data(), data[2], MPI_DOUBLE, i, PRINT_ALL_ACCUMS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(big[i].c.data(), data[2], MPI_INT, i, PRINT_ALL_ACCUMS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(big[i].r.data(), data[3], MPI_INT, i, PRINT_ALL_ACCUMS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    for (int proc_r = 0; proc_r < proc_ct_sqrt; ++proc_r) {
        int proc_start = proc_r * proc_ct_sqrt * layers;
        int proc_end = proc_start + proc_ct_sqrt * layers;
        //cout << proc_start << " " << proc_end << " " << big[proc_start].rows << endl;
        for (int r = 0; r < big[proc_start].rows; r++) {
            for (int p = proc_start; p < proc_end; p++) {
                int prev_col = 0;
                for (int c = big[p].r[r]; c < big[p].r[r + 1]; c++) {
                    for (; prev_col < big[p].c[c]; prev_col++)
                        printf("0 ");
                    prev_col = big[p].c[c] + 1;
                    printf("%.0lf ", big[p].v[c]);
                }
                for (; prev_col < big[p].cols; prev_col++)
                    printf("0 ");
            } 
            puts("");
        }
    }
}

void print_cmp_g(double g) {
    long long x = 0;
    for (auto &v : accum.v)
        x += v > g;
    long long s = 0;
    MPI_Reduce(&x, &s, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (!myRank)
        printf("%lld\n", s);
}

struct Options {
    string file_a;
    string file_b;
    bool v, g_set;
    double g;
    int t;
    int l;

    Options() {
        file_a = file_b = "";
        v = g_set = 0;
        g = 0;
        t = 0;
        l = 1;
    }
};

Options parse_options(int argc, char **argv) {
    Options opts;
    for (int i = 1; i < argc; i++) {
        if (string(argv[i]) == string("-a")) {
            opts.file_a = argv[i + 1];
            i++;
            continue;
        }
        if (string(argv[i]) == string("-b")) {
            opts.file_b = argv[i + 1];
            i++;
            continue;
        }
        if (string(argv[i]) == string("-v")) {
            opts.v = 1;
            continue;
        }
        if (string(argv[i]) == string("-g")) {
            opts.g = strtod(argv[i + 1], NULL);
            opts.g_set = 1;
            i++;
            continue;
        }
        if (string(argv[i]) == string("-t")) {
            opts.t = 0;
            if (string(argv[i + 1]) == "3D")
                opts.t = 1;
            if (string(argv[i + 1]) == "balanced")
                opts.t = 2;
            i++;
            continue;
        }
        if (string(argv[i]) == "-l") {
            opts.l = atoi(argv[i + 1]);
            i++;
            continue;
        }
    }
    if (!opts.t)
        opts.l = 1;
    return opts;
}

void merge_accums(MPI_Comm &fiber_comm, int layers) {
    CSRM split[layers], rcv[layers];
    int col_boundaries[layers];

    auto get_proc_for_col = [&](long long col) {
        return (layers * col) / accum.cols;
    };

    // copied from calc_local_dims!!! FIXME
    auto get_first = [&](long long id) {
        long long p = 0, k = accum.cols - 1, mid;
        while (p <= k) {
            mid = (p + k) >> 1;
            long long x = get_proc_for_col(mid);
            if (x < id) {
                ++p;
                mid = p;
            } else {
                mid = k;
                k--;
            }
        }
        return mid;
    };
    
    for (int i = 0; i < layers; i++) {
        col_boundaries[i] = get_first(i + 1);
        int sr = accum.rows, sc = col_boundaries[i] - (i ? col_boundaries[i - 1] : 0);
        if (MERGE_ACCUMS_DBG) dbg << i << " split " << get_first(i + 1) << " dims " << get_first(i) << endl;
        split[i].set_rowscols(vector<int>({sr, sc}).data());
    }

    if (MERGE_ACCUMS_DBG) {
        dbg << accum.rows << " final frontier " << accum.cols << endl;
        debug_vect(accum.r);
        debug_vect(accum.c);
        debug_vect(accum.v);
    }

    for (int row_nr = 0; row_nr < accum.rows; row_nr++) {
        for (auto &s : split)
            s.r.push_back(s.r.back());

        for (int i = accum.r[row_nr]; i < accum.r[row_nr + 1]; i++) {
            long long id = get_proc_for_col(accum.c[i]);
            if (MERGE_ACCUMS_DBG) dbg << id << " " << row_nr << " " << accum.r[row_nr] << " " << accum.r[row_nr + 1] 
                << "   " << accum.c[i] << " " << i << " " << accum.c.size() << endl;
            split[id].v.push_back(accum.v[i]);
            split[id].c.push_back(accum.c[i] - (id ? col_boundaries[id - 1] : 0));
            split[id].r.back()++;
        }
    }

    if (MERGE_ACCUMS_DBG) {
        dbg << "my accum is: " << endl;
        dbg << accum.rows << " " << accum.cols << endl;
        debug_vect(accum.v); debug_vect(accum.c); debug_vect(accum.r);
    

        dbg << endl << "my split is: " << endl;
        for (int i = 0; i < layers; i++) {
            dbg << i << " " << split[i].rows << " " << split[i].cols << endl;
            debug_vect(split[i].v); debug_vect(split[i].c); debug_vect(split[i].r);
            dbg << endl;
        }

        dbg << "done for now" << endl;
    }
    
    MPI_Barrier(fiber_comm);

    MPI_Request data_requests[2], requests[6];
    MPI_Status data_statuses[2] = {get_new_status(), get_new_status()}, statuses[6];
    for (int i = 0; i < 6; i++)
        statuses[i] = get_new_status();

    for (int r = 0; r < layers; r++) {
        int data[3] = {0, 0, 0}, dn = 3;
        int nxt = (myLayer + r) % layers, prv = (myLayer - r + layers) % layers;
        int csize = (int)split[nxt].c.size();
        int data_sent[3] = {split[nxt].rows, split[nxt].cols, csize};
        MPI_Isend(data_sent, dn, MPI_INT, nxt, 
                    FAKE_ALL_TO_ALL, fiber_comm, &data_requests[0]);
        MPI_Irecv(data, dn, MPI_INT, prv, FAKE_ALL_TO_ALL, fiber_comm, &data_requests[1]);    
        MPI_Waitall(2, data_requests, data_statuses);   

        if (MERGE_ACCUMS_DBG) dbg << data[0] << " " << data[1] << " " << data[2] << endl;
        rcv[prv].set_rowscols(data);
        rcv[prv].v = vector<double> (data[2], 0);
        rcv[prv].c = vector<int> (data[2], 0);
        rcv[prv].r = vector<int> (accum.rows + 1, 0);

        MPI_Isend(split[nxt].v.data(), csize, MPI_DOUBLE, nxt, FAKE_ALL_TO_ALL, fiber_comm, &requests[0]);
        MPI_Irecv(rcv[prv].v.data(), data[2], MPI_DOUBLE, prv, FAKE_ALL_TO_ALL, fiber_comm, &requests[3]);
        MPI_Isend(split[nxt].c.data(), csize, MPI_INT, nxt, FAKE_ALL_TO_ALL, fiber_comm, &requests[1]);
        MPI_Irecv(rcv[prv].c.data(), data[2], MPI_INT, prv, FAKE_ALL_TO_ALL, fiber_comm, &requests[4]);
        MPI_Isend(split[nxt].r.data(), accum.rows + 1, MPI_INT, nxt, FAKE_ALL_TO_ALL, fiber_comm, &requests[2]);
        MPI_Irecv(rcv[prv].r.data(), accum.rows + 1, MPI_INT, prv, FAKE_ALL_TO_ALL, fiber_comm, &requests[5]);
        MPI_Waitall(6, requests, statuses);
                            
    }
    
    if (MERGE_ACCUMS_DBG)
    for (int r = 0; r < layers; r++) {
        dbg << r << " " << rcv[r].rows << " " << rcv[r].cols << endl;
        debug_vect(rcv[r].v);
        debug_vect(rcv[r].c);
        debug_vect(rcv[r].r);
        dbg << "aha\n" << endl;
    }
    
    for (int r = 1; r < layers; r *= 2) {
        for(int i = 0; i < layers - r; i += 2 * r) {
            rcv[i] += move(rcv[i + r]);
        }
    }

    if (MERGE_ACCUMS_DBG) {
        debug_vect(rcv[0].v);
        debug_vect(rcv[0].c);
        debug_vect(rcv[0].r);
    }

    swap(accum, rcv[0]);
    
    MPI_Barrier(fiber_comm);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    double startTime = MPI_Wtime();

    auto opts = parse_options(argc, argv);
    
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    myLayer = myRank % opts.l;
    layerOrder = myRank / opts.l;
    dbg.open((string("debug/") + to_string(myRank)).c_str(), fstream::out);
    
    while (proc_ct_sqrt * proc_ct_sqrt != numProcesses / opts.l) {
        ++proc_ct_sqrt;
        if (proc_ct_sqrt * proc_ct_sqrt * opts.l > numProcesses) {
            printf("mismatching proc counts\n");
            exit(1);
        }
    }
    
    double step1Time = MPI_Wtime();
    
    if (!myRank) {
        bool is_a = true;
        read(opts.file_a, proc_ct_sqrt, opts.l, is_a);

        is_a = false;
        read(opts.file_b, proc_ct_sqrt, opts.l, is_a);
    } else {
        bool is_a = true;
        receive_matrix_dims(is_a);
        receive_matrix(is_a);

        is_a = false;
        receive_matrix_dims(is_a);
        receive_matrix(is_a);
    }
    
    double step2Time = MPI_Wtime();
    
    if (!opts.t)
        calc_local_dims();
    else
        calc_local_dims(opts.l);

    
    double step3Time = MPI_Wtime();
    
    /*dbg << "A" << endl;
    debug_vect(a.v);
    debug_vect(a.c);
    debug_vect(a.r);
    dbg << "B" << endl;
    debug_vect(b.v);
    debug_vect(b.c);
    debug_vect(b.r);
    dbg << "ok" << endl;*/
    b.to_rsc();

    double step4Time = MPI_Wtime();
    
    if (myRank == 0 && numProcesses > 1) {
        MPI_Send(nullptr, 0, MPI_INT, 1, SYNC_AFTER_INITIAL_DISTR, MPI_COMM_WORLD);
        MPI_Recv(nullptr, 0, MPI_INT, numProcesses - 1, SYNC_AFTER_INITIAL_DISTR, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (myRank) {
        MPI_Recv(nullptr, 0, MPI_INT, myRank - 1, SYNC_AFTER_INITIAL_DISTR, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(nullptr, 0, MPI_INT, (myRank + 1) % numProcesses, SYNC_AFTER_INITIAL_DISTR, MPI_COMM_WORLD);
    }
    
    double step5Time = MPI_Wtime();
    
    if (opts.t == 0) {
        summa_2d(MPI_COMM_WORLD, 1);
        //printf("process %d out of %d ok\n", myRank, numProcesses);
    } else {
        MPI_Comm layer_comm, fiber_comm;
        MPI_Comm_split(MPI_COMM_WORLD, myLayer, layerOrder, &layer_comm);
        MPI_Comm_split(MPI_COMM_WORLD, layerOrder, myLayer, &fiber_comm);
        summa_2d(layer_comm, opts.l);
        // b was transposed.
        accum.set_rowscols(vector<int>({a.rows, b.rows}).data());
        merge_accums(fiber_comm, opts.l);
        MPI_Comm_free(&layer_comm);
        MPI_Comm_free(&fiber_comm);
    }

    double step6Time = MPI_Wtime();

    if (opts.v) {
        print_all(opts.l);
    }

    double step7Time = MPI_Wtime();

    if (opts.g_set) {
        print_cmp_g(opts.g);
    }

    double step8Time = MPI_Wtime();

    if (!myRank && 0)
    printf(
        "myRank - %d, times:\ninit - %f\ndistro - %f\n"
        "local dims - %f\nb to rsc - %f\nsync after initial - %f\nSUMMA - %f\n"
        "V - %f\nG - %f\ntotal - %f\nafter distro - %f\n", myRank, step1Time - startTime, step2Time - startTime,
        step3Time - step2Time, step4Time - step3Time, step5Time - step4Time,
        step6Time - step5Time, step7Time - step6Time, step8Time - step7Time, step8Time - startTime, step8Time - step2Time
    );

    MPI_Finalize();
}
