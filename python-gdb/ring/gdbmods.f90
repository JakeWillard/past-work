

module pde
  implicit none

  ! gmres parameters
  real(8) :: error_threshold = 1e-8
  integer :: mi = 50   ! number of gmres iterations before restart
  integer :: mr = 10   ! max number of gmres restarts

  ! differential operators
  real(8), dimension(:,:), allocatable :: partial_x           ! partial with respect to x
  real(8), dimension(:,:), allocatable :: partial_xx
  real(8), dimension(:,:), allocatable :: partial_xy
  real(8), dimension(:,:), allocatable :: partial_y
  real(8), dimension(:,:), allocatable :: partial_yy
  real(8), dimension(:,:), allocatable :: partial_c


  ! field-line tracing and reflection operators
  real(8), dimension(:,:), allocatable :: TR1
  real(8), dimension(:,:), allocatable :: TR2
  real(8), dimension(:,:,:), allocatable :: REF

  ! penalization vectors
  real(8), dimension(:,:), allocatable :: PEN

  ! dimensions of arrays
  integer :: nnz_px, nnz_pxx, nnz_pxy, nnz_py, nnz_pyy, nnz_pc        ! number of nonzero elements for differential operators
  integer :: nnz_TR1, nnz_TR2                                         ! number of nonzero elements for field-line tracing
  integer :: n_bdry                                                   ! number of different boundary surfaces
  integer, dimension(:), allocatable :: nnz_REF                       ! ith element is number of nonzero elements for REF(:,:,i)

  ! grid spacing
  real(8) :: dx, alpha
  real(8), dimension(:), allocatable :: ds1, ds2


contains


  ! LOAD MODULE VARIABLES
  ! =====================

  subroutine load_pde_args()

    use hdf5
    ! character(len=:), allocatable :: filename
    character(len=7) :: filename = "test.h5"

    integer(HID_T) :: file_id, group_id, dset_id, dspace_id ! handles
    integer(HSIZE_T), dimension(3) :: dims, maxdims
    integer :: hdferr

    real(8), dimension(2) :: spacing_vector

    ! start hdf5 interface and open file
    call h5open_f(hdferr)
    call h5fopen_f(filename, H5F_ACC_RDONLY_F, file_id, hdferr)

    ! open "pde" group
    call h5gopen_f(file_id, "/pde", group_id, hdferr)

    ! read grid spacing
    dims(1) = 2
    call h5dopen_f(group_id, "spacing_vector", dset_id, hdferr)
    call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, spacing_vector, dims(1:1), hdferr)
    call h5dclose_f(dset_id, hdferr)
    dx = spacing_vector(1)
    alpha = spacing_vector(2)

    ! read ds1
    call h5dopen_f(group_id, "ds1", dset_id, hdferr)
    call h5dget_space_f(dset_id, dspace_id, hdferr)
    call h5sget_simple_extent_dims_f(dspace_id, dims, maxdims, hdferr)
    allocate(ds1(dims(1)))
    call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, ds1, dims(1:1), hdferr)
    call h5dclose_f(dset_id, hdferr)

    ! read ds2
    call h5dopen_f(group_id, "ds2", dset_id, hdferr)
    allocate(ds2(dims(1)))
    call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, ds2, dims(1:1), hdferr)
    call h5dclose_f(dset_id, hdferr)

    ! read PEN
    call h5dopen_F(group_id, "PEN", dset_id, hdferr)
    call h5dget_space_f(dset_id, dspace_id, hdferr)
    call h5sget_simple_extent_dims_f(dspace_id, dims, maxdims, hdferr)
    n_bdry = dims(2)
    allocate(PEN(dims(1), n_bdry))
    call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, PEN, dims(1:2), hdferr)
    call h5dclose_f(dset_id, hdferr)

    ! read nnz_REF
    dims(1) = n_bdry
    call h5dopen_f(group_id, "nnz_REF", dset_id, hdferr)
    allocate(nnz_REF(n_bdry))
    call h5dread_f(dset_id, H5T_NATIVE_INTEGER, nnz_REF, dims(1:1), hdferr)
    call h5dclose_f(dset_id, hdferr)

    ! open REF
    call h5dopen_F(group_id, "REF", dset_id, hdferr)
    call h5dget_space_f(dset_id, dspace_id, hdferr)
    call h5sget_simple_extent_dims_f(dspace_id, dims, maxdims, hdferr)
    allocate(REF(dims(1), dims(2), dims(3)))
    call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, REF, dims, hdferr)
    call h5dclose_f(dset_id, hdferr)

    ! open TR1
    call h5dopen_F(group_id, "TR1", dset_id, hdferr)
    call h5dget_space_f(dset_id, dspace_id, hdferr)
    call h5sget_simple_extent_dims_f(dspace_id, dims, maxdims, hdferr)
    nnz_TR1 = dims(2)
    allocate(TR1(3,nnz_TR1))
    call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, TR1, dims(1:2), hdferr)
    call h5dclose_f(dset_id, hdferr)

    ! open TR2
    call h5dopen_F(group_id, "TR2", dset_id, hdferr)
    call h5dget_space_f(dset_id, dspace_id, hdferr)
    call h5sget_simple_extent_dims_f(dspace_id, dims, maxdims, hdferr)
    nnz_TR2 = dims(2)
    allocate(TR2(3,nnz_TR2))
    call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, TR2, dims(1:2), hdferr)
    call h5dclose_f(dset_id, hdferr)

    ! open partial_x
    call h5dopen_F(group_id, "partial_x", dset_id, hdferr)
    call h5dget_space_f(dset_id, dspace_id, hdferr)
    call h5sget_simple_extent_dims_f(dspace_id, dims, maxdims, hdferr)
    nnz_px = dims(2)
    allocate(partial_x(3,nnz_px))
    call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, partial_x, dims(1:2), hdferr)
    call h5dclose_f(dset_id, hdferr)

    ! open partial_xx
    call h5dopen_F(group_id, "partial_xx", dset_id, hdferr)
    call h5dget_space_f(dset_id, dspace_id, hdferr)
    call h5sget_simple_extent_dims_f(dspace_id, dims, maxdims, hdferr)
    nnz_pxx = dims(2)
    allocate(partial_xx(3,nnz_pxx))
    call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, partial_xx, dims(1:2), hdferr)
    call h5dclose_f(dset_id, hdferr)

    ! open partial_xy
    call h5dopen_F(group_id, "partial_xy", dset_id, hdferr)
    call h5dget_space_f(dset_id, dspace_id, hdferr)
    call h5sget_simple_extent_dims_f(dspace_id, dims, maxdims, hdferr)
    nnz_pxy = dims(2)
    allocate(partial_xy(3,nnz_pxy))
    call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, partial_xy, dims(1:2), hdferr)
    call h5dclose_f(dset_id, hdferr)

    ! open partial_y
    call h5dopen_F(group_id, "partial_y", dset_id, hdferr)
    call h5dget_space_f(dset_id, dspace_id, hdferr)
    call h5sget_simple_extent_dims_f(dspace_id, dims, maxdims, hdferr)
    nnz_py = dims(2)
    allocate(partial_y(3,nnz_py))
    call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, partial_y, dims(1:2), hdferr)
    call h5dclose_f(dset_id, hdferr)

    ! open partial_yy
    call h5dopen_F(group_id, "partial_yy", dset_id, hdferr)
    call h5dget_space_f(dset_id, dspace_id, hdferr)
    call h5sget_simple_extent_dims_f(dspace_id, dims, maxdims, hdferr)
    nnz_pyy = dims(2)
    allocate(partial_yy(3,nnz_pyy))
    call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, partial_yy, dims(1:2), hdferr)
    call h5dclose_f(dset_id, hdferr)

    ! open partial_c
    call h5dopen_F(group_id, "partial_c", dset_id, hdferr)
    call h5dget_space_f(dset_id, dspace_id, hdferr)
    call h5sget_simple_extent_dims_f(dspace_id, dims, maxdims, hdferr)
    nnz_pc = dims(2)
    allocate(partial_c(3,nnz_pc))
    call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, partial_c, dims(1:2), hdferr)
    call h5dclose_f(dset_id, hdferr)

    ! close the file
    call h5gclose_f(group_id, hdferr)
    call h5fclose_f(file_id, hdferr)
    call h5close_f(hdferr)


  end subroutine load_pde_args


  ! VOLUME PENALIZATION
  ! ===================

  subroutine penalize(i, val, val_b, nxy)

    integer :: i, nxy
    real(8), dimension(nxy) :: val, val_b

    val = (1 - PEN(:,i))*val + PEN(:,i)*val_b

  end subroutine



  ! SPARSE MATRIX MULTIPLICATION
  function sparse_matmul(A, x, n, nnz) result(out)

    integer :: n, nnz, i, j, k
    real(8) :: matrix_element
    real(8), dimension(n) :: x, out
    real(8), dimension(3, nnz) :: A

    ! initialize output
    out = 0

    ! add up result of all nonzero elements
    do k=1,nnz,1
      matrix_element = A(1,k)
      i = INT(A(2,k))
      j = INT(A(3,k))

      out(i) = out(i) + matrix_element*x(j)
    end do

  end function sparse_matmul


  ! PARTIAL DERIVATIVES
  ! ===================


  function D_x(f, nxy)

    integer :: nxy
    real(8), dimension(nxy) :: f, D_x

    D_x = sparse_matmul(partial_x, f, nxy, nnz_px) / dx

  end function D_x


  function D_xx(f, nxy)

    integer :: nxy
    real(8), dimension(nxy) :: f, D_xx

    D_xx = sparse_matmul(partial_xx, f, nxy, nnz_pxx) / dx**2

  end function D_xx


  function D_xy(f, nxy)

    integer :: nxy
    real(8), dimension(nxy) :: f, D_xy

    D_xy = sparse_matmul(partial_xy, f, nxy, nnz_pxy) * alpha / dx**2

  end function D_xy


  function D_y(f, nxy)

    integer :: nxy
    real(8), dimension(nxy) :: f, D_y

    D_y = sparse_matmul(partial_y, f, nxy, nnz_py) * alpha / dx

  end function D_y


  function D_yy(f, nxy)

    integer :: nxy
    real(8), dimension(nxy) :: f, D_yy

    D_yy = sparse_matmul(partial_yy, f, nxy, nnz_pyy) * (alpha/dx)**2

  end function D_yy


  function D_c(f, nxy)

    integer :: nxy
    real(8), dimension(nxy) :: f, D_c

    D_c = sparse_matmul(partial_c, f, nxy, nnz_pc) / dx

  end function D_c


  ! FIELD-LINE TRACE AND BOUNDARY REFLECTION INTERPOLATIONS
  ! =======================================================


  function ftrace(v, nxy)

    integer :: nxy
    real(8), dimension(nxy) :: v, ftrace

    ftrace = sparse_matmul(TR1, v, nxy, nnz_TR1)

  end function ftrace


  function btrace(v, nxy)

    integer :: nxy
    real(8), dimension(nxy) :: v, btrace

    btrace = sparse_matmul(TR2, v, nxy, nnz_TR2)

  end function btrace


  function reflect(v, i, nxy) result(out)

    integer :: nxy, i
    real(8), dimension(nxy) :: v, out

    out = sparse_matmul(REF(:,:,i), v, nxy, nnz_REF(i))

  end function reflect




  ! PARALLEL DERIVATIVES


  function D_s0(f, nxy)

    integer :: nxy
    real(8), dimension(nxy, 3) :: f
    real(8), dimension(nxy) :: D_s0

    real(8), dimension(nxy) :: f1, f2

    f1 = ftrace(f(:,3), nxy)
    f2 = btrace(f(:,1), nxy)

    D_s0 = (f1 - f2) / (ds1 + ds2)

  end function D_s0



  ! GMRES SOLVER
  ! ============


  function gmres_solve(L, x, b, n) result(out)

    ! input args
    real(8), dimension(n) :: x, b
    integer :: n, i, j, k

    interface
      function L(x, n) result(out)
        real(8), dimension(n) :: x, out
      end function L
    end interface

    ! local args
    real(8), dimension(mi,mi+1) :: H        ! hessenberg matrix
    real(8), dimension(n,mi+1) :: Q        ! arnoldi matrix
    real(8), dimension(mi) :: cn     ! givens cosine factors
    real(8), dimension(mi) :: sn     ! givens sine factors
    real(8), dimension(mi+1) :: beta ! projected and rotated residual vector
    real(8), dimension(n) :: r, out
    real(8), dimension(mi) :: y
    real(8) :: temp1, temp2, r_norm, b_norm, error

    ! repeat an mi-iteration cycle until convergence or restarts exceed mr
    do i=1,mr,1

      ! compute initial residual
      r = b - L(x, n)

      ! compute r_norm and b_norm
      r_norm = norm2(r)
      b_norm = norm2(b)

      ! initialize Q and beta for this cycle
      Q(:,1) = r / r_norm
      beta(1) = r_norm

      ! perform gmres iteration
      do k=1,mi,1

        ! apply operator
        Q(:,k+1) = L(Q(:,k), n)

        ! carry out arnoldi process
        do j=1,k,1
          H(j,k) = dot_product(Q(:,k+1), Q(:,j))
          Q(:,k+1) = Q(:,k+1) - H(j,k) * Q(:,j)
        end do
        H(k+1,k) = norm2(Q(:,k+1))
        Q(:,k+1) = Q(:,k+1) / H(k+1,k)

        ! carry out givens rotation
        do j=1,k-1,1
          temp1 = cn(j) * H(j,k) + sn(j) * H(j+1,k)
          H(j,k) = -sn(j) * H(j,k) + cn(j) * H(j+1,k)
          H(j+1,k) = temp1
        end do

        ! update cn and sn
        temp2 = sqrt(H(k,k)**2 + H(k+1,k)**2)
        cn(k) = H(k,k) / temp2
        sn(k) = H(k+1,k) / temp2

        ! eliminate H(k+1,k)
        H(k,k) = cn(k) * H(k,k) + sn(k) * H(k+1,k)
        H(k+1,k) = 0

        ! update residual vector
        beta(k+1) = -sn(k) * beta(k)
        beta(k) = cn(k) * beta(k)

        ! compute the error
        error = abs(beta(k+1)) / b_norm
      end do

      ! get y from backward substitution (H(1:mi,1:mi)*y = beta(1:mi))
      do k=mi,1,-1
        y(k) = beta(k) / H(k,k)
        do j=mi,k+1,-1
          y(k) = y(k) - H(k,j)*beta(j) / H(k,k)
        end do
      end do

      ! compute updated x from y
      x = x + matmul(Q(:,1:mi), y)

      ! restart iteration unless error is within threshold
      if (error < error_threshold) then
        exit
      end if
    end do

    ! set output vector to final x
    out = x

  end function gmres_solve


end module pde




module comms

  use mpi
  implicit none

  ! mpi args
  integer :: ierr, rank, nprocs
  integer, dimension(MPI_STATUS_SIZE) :: status1


contains


  subroutine mpi_setup()

    call MPI_INIT(ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

  end subroutine mpi_setup


  subroutine share_vals(x, nxy)

    integer :: nxy, i
    real(8), dimension(nxy, 3) :: x

    ! have rank 0 and rank nprocs-1 share
    if (rank == nprocs-1) then

      call MPI_SEND(x(:,2), nxy, MPI_DOUBLE_PRECISION, 0, 2*(nprocs-1)-1, MPI_COMM_WORLD, ierr)
      call MPI_RECV(x(:,3), nxy, MPI_DOUBLE_PRECISION, 0, 2*(nprocs-1), MPI_COMM_WORLD, status1, ierr)

    else if (rank == 0) then

      call MPI_RECV(x(:,1), nxy, MPI_DOUBLE_PRECISION, nprocs-1, 2*(nprocs-1)-1, MPI_COMM_WORLD, status1, ierr)
      call MPI_SEND(x(:,2), nxy, MPI_DOUBLE_PRECISION, nprocs-1, 2*(nprocs-1), MPI_COMM_WORLD, ierr)

    end if

    ! have all other neighboring processes share
    do i=1,nprocs-2,1

      if (rank == i) then

        call MPI_SEND(x(:,2), nxy, MPI_DOUBLE_PRECISION, i+1, 2*i-1, MPI_COMM_WORLD, ierr)
        call MPI_RECV(x(:,3), nxy, MPI_DOUBLE_PRECISION, i+1, 2*i, MPI_COMM_WORLD, status1, ierr)

      else if (rank == i+1) then

        call MPI_RECV(x(:,1), nxy, MPI_DOUBLE_PRECISION, i, 2*i-1, MPI_COMM_WORLD, status1, ierr)
        call MPI_SEND(x(:,2), nxy, MPI_DOUBLE_PRECISION, i, 2*i, MPI_COMM_WORLD, ierr)

      end if

    end do


  end subroutine share_vals


  function master_collect(x, m) result(out)

    integer :: m, i
    real(8), dimension(m, nprocs) :: out
    real(8), dimension(m) :: x

    ! send master process data that master process doesn't already have
    do i=1,nprocs-1,1
      if (rank == i) then
        call MPI_SEND(x(:), m, MPI_DOUBLE_PRECISION, 0, i, MPI_COMM_WORLD, ierr)
      end if
    end do

    ! master process produces array out with all of the data
    if (rank == 0) then

      out(:,1) = x(:)
      do i=1,nprocs-1,1
        call MPI_RECV(out(:,i+1), m, MPI_DOUBLE_PRECISION, i, i, MPI_COMM_WORLD, status1, ierr)
      end do

    end if

  end function master_collect



end module comms



module model

  use pde
  use comms
  implicit none

  ! declare physical parameters
  real(8) :: am, ad, ke, ki, er, eg, ev, de2, eta

  ! floating potential
  real(8) :: lambda = 2.6

  ! flux surface average kernal
  real(8), dimension(:), allocatable :: flx_avg

  ! declare grid size
  integer :: nxy

  ! declare variables with explicit time evolution
  real(8), dimension(:,:,:), allocatable :: lnn     ! natural log of density
  real(8), dimension(:,:,:), allocatable :: lnTi    ! natural log of ion temperature
  real(8), dimension(:,:,:), allocatable :: lnTe    ! natural log of electron temperature
  real(8), dimension(:,:,:), allocatable :: vp      ! ion parallel velocity
  real(8), dimension(:,:,:), allocatable :: w       ! vorticity
  real(8), dimension(:,:,:), allocatable :: psihat  ! shifted flux function (psi - de*j)

  ! declare implicitely evolving variables
  real(8), dimension(:,:), allocatable :: n       ! density
  real(8), dimension(:,:), allocatable :: Ti      ! ion temperature
  real(8), dimension(:,:), allocatable :: Te      ! electron temperature
  real(8), dimension(:,:), allocatable :: j       ! parallel current
  real(8), dimension(:,:), allocatable :: psi     ! variable flux function
  real(8), dimension(:,:), allocatable :: phi     ! electric potential
  real(8), dimension(:,:), allocatable :: Pe      ! electron pressure
  real(8), dimension(:,:), allocatable :: Pi      ! ion pressure
  real(8), dimension(:,:), allocatable :: G       ! pumping term
  real(8), dimension(:,:), allocatable :: E1      ! x-component of del*phi + ad*del*Pi/n
  real(8), dimension(:,:), allocatable :: E2      ! y-component of del*phi + ad*del*Pi/n
  real(8), dimension(:,:), allocatable :: jn      ! current divided by density

  ! declare x derivatives of variables
  real(8), dimension(:,:), allocatable :: lnn_x
  real(8), dimension(:,:), allocatable :: lnTi_x
  real(8), dimension(:,:), allocatable :: lnTe_x
  real(8), dimension(:,:), allocatable :: vp_x
  real(8), dimension(:,:), allocatable :: j_x
  real(8), dimension(:,:), allocatable :: jn_x
  real(8), dimension(:,:), allocatable :: Pe_x
  real(8), dimension(:,:), allocatable :: Pi_x
  real(8), dimension(:,:), allocatable :: Pi_xx
  real(8), dimension(:,:), allocatable :: G_x
  real(8), dimension(:,:), allocatable :: E1_x
  real(8), dimension(:,:), allocatable :: E2_x
  real(8), dimension(:,:), allocatable :: E1_xx
  real(8), dimension(:,:), allocatable :: E2_xx
  real(8), dimension(:,:), allocatable :: psi_x
  real(8), dimension(:,:), allocatable :: phi_x
  real(8), dimension(:,:), allocatable :: phi_xx

  ! declare y derivatives of variables
  real(8), dimension(:,:), allocatable :: lnn_y
  real(8), dimension(:,:), allocatable :: lnTi_y
  real(8), dimension(:,:), allocatable :: lnTe_y
  real(8), dimension(:,:), allocatable :: vp_y
  real(8), dimension(:,:), allocatable :: j_y
  real(8), dimension(:,:), allocatable :: jn_y
  real(8), dimension(:,:), allocatable :: Pe_y
  real(8), dimension(:,:), allocatable :: Pi_y
  real(8), dimension(:,:), allocatable :: Pi_yy
  real(8), dimension(:,:), allocatable :: G_y
  real(8), dimension(:,:), allocatable :: E1_y
  real(8), dimension(:,:), allocatable :: E2_y
  real(8), dimension(:,:), allocatable :: E1_yy
  real(8), dimension(:,:), allocatable :: E2_yy
  real(8), dimension(:,:), allocatable :: psi_y
  real(8), dimension(:,:), allocatable :: phi_y
  real(8), dimension(:,:), allocatable :: phi_yy

  ! mixed derivatives
  real(8), dimension(:,:), allocatable :: phi_xy
  real(8), dimension(:,:), allocatable :: E1_xy
  real(8), dimension(:,:), allocatable :: E2_xy

  ! declare s derivatives of variables
  real(8), dimension(:,:), allocatable :: lnn_s
  real(8), dimension(:,:), allocatable :: j_s
  real(8), dimension(:,:), allocatable :: jn_s
  real(8), dimension(:,:), allocatable :: vp_s
  real(8), dimension(:,:), allocatable :: lnTi_s
  real(8), dimension(:,:), allocatable :: lnTe_s
  real(8), dimension(:,:), allocatable :: lnTi_ss
  real(8), dimension(:,:), allocatable :: lnTe_ss
  real(8), dimension(:,:), allocatable :: Pe_s
  real(8), dimension(:,:), allocatable :: Pi_s
  real(8), dimension(:,:), allocatable :: G_s
  real(8), dimension(:,:), allocatable :: phi_s

  ! declare curvature operator applied to variables
  real(8), dimension(:,:), allocatable :: Pe_c
  real(8), dimension(:,:), allocatable :: Pi_c
  real(8), dimension(:,:), allocatable :: lnTe_c
  real(8), dimension(:,:), allocatable :: lnTi_c
  real(8), dimension(:,:), allocatable :: phi_c
  real(8), dimension(:,:), allocatable :: vp_c
  real(8), dimension(:,:), allocatable :: G_c


contains


  ! LOAD MODULE VARIABLES
  ! =====================

  subroutine load_model_args(filename)

    use hdf5
    character(len=:), allocatable :: filename
    real(8), dimension(9) :: params

    integer(HID_T) :: file_id, group_id, dset_id, dspace_id ! handles
    integer(HSIZE_T), dimension(3) :: dims, maxdims
    integer(HSIZE_T), dimension(3) :: start, stride, bloc, count
    integer :: hdferr

    ! start hdf5 interface and open file
    call h5open_f(hdferr)
    call h5fopen_f(filename, H5F_ACC_RDONLY_F, file_id, hdferr)

    ! open "model" group
    call h5gopen_f(file_id, "/model", group_id, hdferr)

    ! read params
    dims(1) = 9
    call h5dopen_f(group_id, "params", dset_id, hdferr)
    call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, params, dims(1:1), hdferr)
    call h5dclose_f(dset_id, hdferr)
    am = params(1)
    ad = params(2)
    ke = params(3)
    ki = params(4)
    er = params(5)
    eg = params(6)
    ev = params(7)
    de2 = params(8)
    eta = params(9)

    ! open lnn, get dimensions and hyperslab for reading initial values
    call h5dopen_F(group_id, "lnn", dset_id, hdferr)
    call h5dget_space_f(dset_id, dspace_id, hdferr)
    call h5sget_simple_extent_dims_f(dspace_id, dims, maxdims, hdferr)
    nxy = dims(1)
    start = (/0,rank,0/)
    count = (/1,1,1/)
    stride = (/10,10,10/) ! we are just getting one slab so this shouldn't even matter?
    bloc = (/nxy, 1, 1/)
    call h5sselect_hyperslab_f(dspace_id, H5S_SELECT_SET_F, start, count, hdferr, stride, bloc)

    ! allocate explicitely evolving variables
    allocate(lnn(nxy,3,3), lnTe(nxy,3,3), lnTi(nxy,3,3), vp(nxy,3,3), w(nxy,3,3), psihat(nxy,3,3))

    ! allocate flx_avg
    allocate(flx_avg(nxy))

    ! allocate everything else
    allocate(n(nxy,3), Ti(nxy,3), Te(nxy,3), j(nxy,3), psi(nxy,3), &
             phi(nxy,3), Pe(nxy,3), Pi(nxy,3), G(nxy,3), E1(nxy,3), &
             E2(nxy,3), jn(nxy,3), lnn_x(nxy,3), lnTi_x(nxy,3), &
             lnTe_x(nxy,3), vp_x(nxy,3), j_x(nxy,3), jn_x(nxy,3), &
             Pe_x(nxy,3), Pi_x(nxy,3), Pi_xx(nxy,3), G_x(nxy,3), &
             E1_x(nxy,3), E2_x(nxy,3), E1_xx(nxy,3), E2_xx(nxy,3), &
             psi_x(nxy,3), phi_x(nxy,3), phi_xx(nxy,3), lnn_y(nxy,3), &
             lnTi_y(nxy,3), lnTe_y(nxy,3), vp_y(nxy,3), j_y(nxy,3), &
             jn_y(nxy,3), Pe_y(nxy,3), Pi_y(nxy,3), Pi_yy(nxy,3), &
             G_y(nxy,3), E1_y(nxy,3), E2_y(nxy,3), E1_yy(nxy,3), &
             E2_yy(nxy,3), psi_y(nxy,3), phi_y(nxy,3), phi_yy(nxy,3), &
             phi_xy(nxy,3), E1_xy(nxy,3), E2_xy(nxy,3), lnn_s(nxy,3), &
             j_s(nxy,3), jn_s(nxy,3), vp_s(nxy,3), lnTi_s(nxy,3), &
             lnTe_s(nxy,3), lnTi_ss(nxy,3), lnTe_ss(nxy,3), Pe_s(nxy,3), &
             Pi_s(nxy,3), G_s(nxy,3), phi_s(nxy,3), Pe_c(nxy,3), Pi_c(nxy,3), &
             lnTe_c(nxy,3), lnTi_c(nxy,3), phi_c(nxy,3), vp_c(nxy,3), G_c(nxy,3))

    ! read initial values
    call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, lnn(:,2,1), dims(1:1), hdferr, file_space_id=dspace_id)
    call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, lnn(:,2,2), dims(1:1), hdferr, file_space_id=dspace_id)
    call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, lnn(:,2,3), dims(1:1), hdferr, file_space_id=dspace_id)
    call h5dclose_f(dset_id, hdferr)

    call h5dopen_f(group_id, "lnTe", dset_id, hdferr)
    call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, lnTe(:,2,1), dims(1:1), hdferr, file_space_id=dspace_id)
    call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, lnTe(:,2,2), dims(1:1), hdferr, file_space_id=dspace_id)
    call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, lnTe(:,2,3), dims(1:1), hdferr, file_space_id=dspace_id)
    call h5dclose_f(dset_id, hdferr)

    call h5dopen_f(group_id, "lnTi", dset_id, hdferr)
    call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, lnTi(:,2,1), dims(1:1), hdferr, file_space_id=dspace_id)
    call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, lnTi(:,2,2), dims(1:1), hdferr, file_space_id=dspace_id)
    call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, lnTi(:,2,3), dims(1:1), hdferr, file_space_id=dspace_id)
    call h5dclose_f(dset_id, hdferr)

    call h5dopen_f(group_id, "vp", dset_id, hdferr)
    call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, vp(:,2,1), dims(1:1), hdferr, file_space_id=dspace_id)
    call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, vp(:,2,2), dims(1:1), hdferr, file_space_id=dspace_id)
    call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, vp(:,2,3), dims(1:1), hdferr, file_space_id=dspace_id)
    call h5dclose_f(dset_id, hdferr)

    call h5dopen_f(group_id, "w", dset_id, hdferr)
    call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, w(:,2,1), dims(1:1), hdferr, file_space_id=dspace_id)
    call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, w(:,2,2), dims(1:1), hdferr, file_space_id=dspace_id)
    call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, w(:,2,3), dims(1:1), hdferr, file_space_id=dspace_id)
    call h5dclose_f(dset_id, hdferr)

    call h5dopen_f(group_id, "psihat", dset_id, hdferr)
    call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, psihat(:,2,1), dims(1:1), hdferr, file_space_id=dspace_id)
    call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, psihat(:,2,2), dims(1:1), hdferr, file_space_id=dspace_id)
    call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, psihat(:,2,3), dims(1:1), hdferr, file_space_id=dspace_id)
    call h5dclose_f(dset_id, hdferr)

    ! read flx_avg
    call h5dopen_f(group_id, "flx_avg", dset_id, hdferr)
    call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, flx_avg, dims(1:1), hdferr)
    call h5dclose_f(dset_id, hdferr)

    ! close the file
    call h5gclose_f(group_id, hdferr)
    call h5fclose_f(file_id, hdferr)
    call h5close_f(hdferr)

  end subroutine load_model_args


  ! ELEMENTARY PENALIZATIONS
  ! ========================

  subroutine neumann_penalization(val, val_0)

    real(8), dimension(nxy) :: val
    real(8), dimension(nxy, 3) :: val_0

    ! core and wall boundaries via reflection
    call penalize(1, val, val_0(:,2) - reflect(val_0(:,2), 1, nxy), nxy)
    call penalize(2, val, val_0(:,2) - reflect(val_0(:,2), 2, nxy), nxy)

    ! use field-line maps for plates
    if (n_bdry > 2) then
      call penalize(3, val, val_0(:,2) - btrace(val_0(:,1), nxy), nxy)
      call penalize(4, val, val_0(:,2) - ftrace(val_0(:,3), nxy), nxy)
    end if

    ! reflection again on dome surface
    if (n_bdry > 4) then
      call penalize(5, val, val_0(:,2) - reflect(val_0(:,2), 5, nxy), nxy)
    end if

  end subroutine neumann_penalization


  subroutine dirichlet_penalization(val, val_0)

    integer :: i
    real(8), dimension(nxy) :: val, val_0

    ! set value to minus reflection across all boundaries
    do i=1,n_bdry,1
      call penalize(i, val, val_0 + reflect(val_0, i, nxy), nxy)
    end do

  end subroutine dirichlet_penalization



  ! TIME DERIVATIVES
  ! ================


  function lnn_t(t)

    integer :: t
    real(8), dimension(nxy) :: lnn_t

    lnn_t = er*(phi_c(:,2) - ad*(Pe_c(:,2) &
            + j_s(:,2))/n(:,2)) - ev*vp_s(:,2) &
            -(phi_y(:,2)*lnn_x(:,2) - phi_x(:,2)*lnn_y(:,2))

    ! neumann boundary conditions
    call neumann_penalization(lnn_t, lnn(:,:,t))

  end function lnn_t


  function lnTe_t(t)

    integer :: t
    real(8), dimension(nxy) :: lnTe_t

    lnTe_t = (2/3)*(lnn_t(t) + (phi_y(:,2)*lnn_x(:,2) &
             - phi_x(:,2)*lnn_y(:,2) + er*ad*j(:,2) &
             *lnn_s(:,2)/n(:,2))) - (5/3)*er*ad*Te(:,2)*lnTe_c(:,2) &
             + (ke*Te(:,2)**(5/2)/n(:,2)) * ((7/2)*lnTe_s(:,2)**2 &
             + lnTe_ss(:,2)) - (phi_y(:,2)*lnTe_x(:,2) &
             - phi_x(:,2)*lnTe_y(:,2))

    ! neumann boundary conditions
    call neumann_penalization(lnTe_t, lnTe(:,:,t))

  end function lnTe_t


  function lnTi_t(t)

    integer :: t
    real(8), dimension(nxy) :: lnTi_t

    lnTi_t = (2/3)*(lnn_t(t) + (phi_y(:,2)*lnn_x(:,2) - phi_x(:,2)*lnn_y(:,2))) &
             + (5/3)*er*ad*Ti(:,2)*lnTi_c(:,2) - (phi_y(:,2)*lnTi_x(:,2) &
             - phi_x(:,2)*lnTi_y(:,2)) + (ki*Ti(:,2)**(5/2)/n(:,2)) &
             * ((7/2)*lnTi_s(:,2)**2 + lnTi_ss(:,2))

    ! neumann boundary conditions
    call neumann_penalization(lnTi_t, lnTi(:,:,t))

  end function lnTi_t


  function vp_t(t)

    integer :: t
    real(8), dimension(nxy) :: vp_t

    vp_t = -(ev/n(:,2))*(Pe_s(:,2) + Pi_s(:,2) + 4*eg*G_s(:,2)) &
           + er*ad*Ti(:,2)*vp_c(:,2) - (phi_y(:,2)*vp_x(:,2) &
           - phi_x(:,2)*vp_y(:,2))

    ! neumann boundary conditions
    call neumann_penalization(vp_t, vp(:,:,t))

  end function vp_t


  function w_t(t)

    integer :: t
    real(8), dimension(nxy) :: w_t

    w_t = (phi_y(:,2)*E1_x(:,2) - phi_x(:,2)*E1_y(:,2))*n(:,2)*lnn_x(:,2) &
        + (phi_y(:,2)*E2_x(:,2) - phi_x(:,2)*E2_y(:,2))*n(:,2)*lnn_y(:,2) &
        + n(:,2)*(phi_xy(:,2)*E1_x(:,2) + phi_y(:,2)*E1_xx(:,2)) &
        - n(:,2)*(phi_xx(:,2)*E1_y(:,2) - phi_x(:,2)*E1_xy(:,2)) &
        + n(:,2)*(phi_yy(:,2)*E2_x(:,2) + phi_y(:,2)*E2_xy(:,2)) &
        - n(:,2)*(phi_xy(:,2)*E2_y(:,2) - phi_x(:,2)*E2_yy(:,2)) &
        + j_s(:,2) - eg*G_c(:,2) - Pe_c(:,2) - Pi_c(:,2)

    ! dirichlet boundary conditions
    call dirichlet_penalization(w_t, w(:,2,t))

  end function w_t


  function psihat_t(t)

    integer :: t
    real(8), dimension(nxy) :: psihat_t

    psihat_t = de2 *(phi_y(:,2)*jn_x(:,2) - phi_x(:,2)*jn_y(:,2) &
             + (vp(:,2,t)-jn(:,2)/ev)*jn_s(:,2)/ev) &
             + (phi_s(:,2) - ad*Pe_s(:,2)/n(:,2))/am + eta*j(:,2)/Te(:,2)**(3/2)

    ! dirichlet boundary conditions
    call dirichlet_penalization(psihat_t, psihat(:,2,t))

  end function psihat_t


  ! LINEAR PROBLEMS
  ! ===============


  subroutine linsolve(x, lhs, rhs)

    real(8), dimension(nxy) :: x, rhs

    interface
      function lhs(y, m) result(out)
        integer :: m
        real(8), dimension(m) :: y, out
      end function lhs
    end interface

    x = gmres_solve(lhs, x, rhs, nxy)

  end subroutine linsolve


  function vorticity_lhs(x, m) result(out)

    integer :: m, i
    real(8), dimension(m) :: x, out
    real(8), dimension(m) :: ones

    out = n(:,2)*(D_xx(x, m) + D_yy(x, m) + lnn_x(:,2)*D_x(x, m) &
          + lnn_y(:,2)*D_y(x, m))

    ! homogenous condition on core-side boundary
    ones = 1
    call penalize(1, out, x - ones*dot_product(flx_avg, x), nxy)

    ! inhomogenous boundary value on all other surfaces
    do i=2,n_bdry,1
      call penalize(i, out, x + reflect(x, i, m), nxy)
    end do

  end function vorticity_lhs


  function vorticity_rhs() result(out)

    integer :: i
    real(8), dimension(nxy) :: out

    ! initialize with values at time t
    out = w(:,2,2) - ad*(Pi_xx(:,2) + Pi_yy(:,2))

    ! penalize rhs for a homogenous condition at core-side boundary
    out = (1 - PEN(:,1))*out

    ! assign sheath boundary value on other surfaces
    do i=2,n_bdry,1
      call penalize(i, out, lambda*(Te(:,2) + reflect(Te(:,2), i, nxy)), nxy)
    end do

  end function vorticity_rhs


  function ohm_lhs(x, m) result(out)

    integer :: m, i
    real(8), dimension(m) :: x, out

    out = x - de2*(D_xx(x, m) + D_yy(x, m))

    ! homogeneous dirichlet conditions on all boundaries
    do i=1,n_bdry,1
      call penalize(i, out, x + reflect(x, i, m), nxy)
    end do

  end function ohm_lhs


  function ohm_rhs() result(out)

    integer :: i
    real(8), dimension(nxy) :: out

    out = psihat(:,2,3)

    do i=1,n_bdry,1
      out = (1 - PEN(:,i))*out
    end do

  end function ohm_rhs


  function diff_n_lhs(x, m) result(out)

    integer :: m, i
    real(8), dimension(m) :: x, out


  end function diff_n_lhs


  function diff_n_rhs() result(out)

    integer :: i
    real(8), dimension(nxy) :: out

  end function diff_n_rhs


  function diff_Te_lhs(x, m) result(out)

    integer :: m, i
    real(8), dimension(m) :: x, out

  end function diff_Te_lhs


  function diff_Te_rhs() result(out)

    integer :: i
    real(8), dimension(nxy) :: out

  end function diff_Te_rhs


  function diff_Ti_lhs(x, m) result(out)

    integer :: m, i
    real(8), dimension(m) :: x, out

  end function diff_Ti_lhs


  function diff_Ti_rhs() result(out)

    integer :: i
    real(8), dimension(nxy) :: out

  end function diff_Ti_rhs


  function diff_vp_lhs(x, m) result(out)

    integer :: m, i
    real(8), dimension(m) :: x, out

  end function diff_vp_lhs


  function diff_vp_rhs() result(out)

    integer :: i
    real(8), dimension(nxy) :: out

  end function diff_vp_rhs


  function diff_w_lhs(x, m) result(out)

    integer :: m, i
    real(8), dimension(m) :: x, out

  end function diff_w_lhs


  function diff_w_rhs() result(out)

    integer :: i
    real(8), dimension(nxy) :: out

  end function diff_w_rhs


  function diff_psihat_lhs(x, m) result(out)

    integer :: m, i
    real(8), dimension(m) :: x, out

  end function diff_psihat_lhs


  function diff_psihat_rhs() result(out)

    integer :: i
    real(8), dimension(nxy) :: out

  end function diff_psihat_rhs


  ! IMPLICIT TIME EVOLUTION
  ! =======================


  subroutine evolve_implicit()

    ! invert logs
    n(:,2) = EXP(lnn(:,2,3))
    Te(:,2) = EXP(lnTe(:,2,3))
    Ti(:,2) = EXP(lnTi(:,2,3))

    ! compute Pe, and Pi
    Pe(:,2) = n(:,2) * Te(:,2)
    Pi(:,2) = n(:,2) * Ti(:,2)

    ! compute phi and psi
    call linsolve(phi(:,2), vorticity_lhs, vorticity_rhs())
    call linsolve(psi(:,2), ohm_lhs, ohm_rhs())

    ! compute j and jn
    j(:,2) = D_xx(psi(:,2), nxy) + D_yy(psi(:,2), nxy)
    jn(:,2) = j(:,2) / n(:,2)

    ! compute x derivatives
    lnn_x(:,2) = D_x(lnn(:,2,3), nxy)
    lnTi_x(:,2) = D_x(lnTi(:,2,3), nxy)
    lnTe_x(:,2) = D_x(lnTe(:,2,3), nxy)
    vp_x(:,2) = D_x(vp(:,2,3), nxy)
    j_x(:,2) = D_x(j(:,2), nxy)
    jn_x(:,2) = D_x(jn(:,2), nxy)
    Pe_x(:,2) = D_x(Pe(:,2), nxy)
    Pi_x(:,2) = D_x(Pi(:,2), nxy)
    Pi_xx(:,2) = D_xx(Pi(:,2), nxy)
    psi_x(:,2) = D_x(psi(:,2), nxy)
    phi_x(:,2) = D_x(phi(:,2), nxy)
    phi_xx(:,2) = D_xx(phi(:,2), nxy)

    ! compute y derivatives
    lnn_y(:,2) = D_y(lnn(:,2,3), nxy)
    lnTi_y(:,2) = D_y(lnTi(:,2,3), nxy)
    lnTe_y(:,2) = D_y(lnTe(:,2,3), nxy)
    vp_y(:,2) = D_y(vp(:,2,3), nxy)
    j_y(:,2) = D_y(j(:,2), nxy)
    jn_y(:,2) = D_y(jn(:,2), nxy)
    Pe_y(:,2) = D_y(Pe(:,2), nxy)
    Pi_y(:,2) = D_y(Pi(:,2), nxy)
    Pi_yy(:,2) = D_yy(Pi(:,2), nxy)
    psi_y(:,2) = D_y(psi(:,2), nxy)
    phi_y(:,2) = D_y(phi(:,2), nxy)
    phi_yy(:,2) = D_yy(phi(:,2), nxy)

    ! mixed derivatives of phi
    phi_xy(:,2) = D_xy(phi(:,2), nxy)

    ! share data for parallel derivatives
    call share_vals(lnn(:,:,3), nxy)
    call share_vals(j(:,:), nxy)
    call share_vals(jn(:,:), nxy)
    call share_vals(vp(:,:,3), nxy)
    call share_vals(lnTi(:,:,3), nxy)
    call share_vals(lnTe(:,:,3), nxy)
    call share_vals(Pe(:,:), nxy)
    call share_vals(Pi(:,:), nxy)
    call share_vals(phi(:,:), nxy)

    ! compute parallel derivatives
    lnn_s(:,2) = D_s0(lnn(:,:,3), nxy) &
                   + psi_y(:,2)*lnn_x(:,2) - psi_x(:,2)*lnn_y(:,2)
    j_s(:,2) = D_s0(j(:,:), nxy) &
                   + psi_y(:,2)*j_x(:,2) - psi_x(:,2)*j_y(:,2)
    jn_s(:,2) = D_s0(jn(:,:), nxy) &
                   + psi_y(:,2)*jn_x(:,2) - psi_x(:,2)*jn_y(:,2)
    vp_s(:,2) = D_s0(vp(:,:,3), nxy) &
                   + psi_y(:,2)*vp_x(:,2) - psi_x(:,2)*vp_y(:,2)
    lnTi_s(:,2) = D_s0(lnTi(:,:,3), nxy) &
                   + psi_y(:,2)*lnTi_x(:,2) - psi_x(:,2)*lnTi_y(:,2)
    lnTe_s(:,2) = D_s0(lnTe(:,:,3), nxy) &
                   + psi_y(:,2)*lnTe_x(:,2) - psi_x(:,2)*lnTe_y(:,2)
    Pe_s(:,2) = D_s0(Pe(:,:), nxy) &
                   + psi_y(:,2)*Pe_x(:,2) - psi_x(:,2)*Pe_y(:,2)
    Pi_s(:,2) = D_s0(Pi(:,:), nxy) &
                   + psi_y(:,2)*Pi_x(:,2) - psi_x(:,2)*Pi_y(:,2)
    phi_s(:,2) = D_s0(phi(:,:), nxy) &
                   + psi_y(:,2)*phi_x(:,2) - psi_x(:,2)*phi_y(:,2)

    ! share lnTe_s and lnTi_s data
    call share_vals(lnTe_s(:,:), nxy)
    call share_vals(lnTi_s(:,:), nxy)

    ! compute second parallel derivatives
    lnTe_ss(:,2) = D_s0(lnTe_s(:,:), nxy) &
                    + psi_y(:,2)*D_x(lnTe_s(:,2), nxy) &
                    - psi_x(:,2)*D_y(lnTe_s(:,2), nxy)
    lnTi_ss(:,2) = D_s0(lnTi_s(:,:), nxy) &
                    + psi_y(:,2)*D_x(lnTi_s(:,2), nxy) &
                    - psi_x(:,2)*D_y(lnTi_s(:,2), nxy)

    ! compute curvature operator
    Pe_c(:,2) = D_c(Pe(:,2), nxy)
    Pi_c(:,2) = D_c(Pi(:,2), nxy)
    lnTe_c(:,2) = D_c(lnTe(:,2,3), nxy)
    lnTi_c(:,2) = D_c(lnTi(:,2,3), nxy)
    phi_c(:,2) = D_c(phi(:,2), nxy)
    vp_c(:,2) = D_c(vp(:,2,3), nxy)

    ! compute G, E1 and E2
    E1(:,2) = phi_x(:,2) + ad*Pi_x(:,2) / n(:,2)
    E2(:,2) = phi_y(:,2) + ad*Pi_y(:,2) / n(:,2)
    G(:,2) = (Ti(:,2)**(5/2))*(er*(phi_c(:,2) &
               + ad*Pi_c(:,2)/n(:,2)) -4*ev*vp_s(:,2))

    ! share G
    call share_vals(G(:,:), nxy)

    ! all derivatives of G, E1, and E2
    E1_x(:,2) = D_x(E1(:,2), nxy)
    E1_xx(:,2) = D_xx(E1(:,2), nxy)
    E2_x(:,2) = D_x(E2(:,2), nxy)
    E2_xx(:,2) = D_xx(E2(:,2), nxy)
    G_x(:,2) = D_x(G(:,2), nxy)
    E1_y(:,2) = D_y(E1(:,2), nxy)
    E1_yy(:,2) = D_yy(E1(:,2), nxy)
    E2_y(:,2) = D_y(E2(:,2), nxy)
    E2_yy(:,2) = D_yy(E2(:,2), nxy)
    G_y(:,2) = D_y(G(:,2), nxy)
    G_c(:,2) = D_c(G(:,2), nxy)
    G_s(:,2) = D_s0(G(:,:), nxy) &
                   + psi_y(:,2)*G_x(:,2) - psi_x(:,2)*G_y(:,2)

  end subroutine evolve_implicit


end module model



module integrators

  use model
  implicit none

  real(8) :: dt = 0.0001


contains


  subroutine leapfrog()

    ! calculate predictor steps
    lnn(:,2,3) = (lnn(:,2,1) + lnn(:,2,2))/2.0 + dt*lnn_t(2)
    lnTi(:,2,3) = (lnTi(:,2,1) + lnTi(:,2,2))/2.0 + dt*lnTi_t(2)
    lnTe(:,2,3) = (lnTe(:,2,1) + lnTe(:,2,2))/2.0 + dt*lnTe_t(2)
    vp(:,2,3) = (vp(:,2,1) + vp(:,2,2))/2.0 + dt*vp_t(2)
    w(:,2,3) = (w(:,2,1) + w(:,2,2))/2.0 + dt*w_t(2)
    psihat(:,2,3) = (psihat(:,2,1) + psihat(:,2,2))/2.0 + dt*psihat_t(2)

    ! fill in all other variables
    call evolve_implicit()

    ! corrector step
    lnn(:,2,3) = lnn(:,2,2) + dt*lnn_t(3)
    lnTi(:,2,3) = lnTi(:,2,2) + dt*lnTi_t(3)
    lnTe(:,2,3) = lnTe(:,2,2) + dt*lnTe_t(3)
    vp(:,2,3) = vp(:,2,2) + dt*vp_t(3)
    w(:,2,3) = w(:,2,2) + dt*w_t(3)
    psihat(:,2,3) = psihat(:,2,2) + dt*psihat_t(3)

    ! fill in other variables again
    call evolve_implicit()

    ! shift the new values to old values
    lnn(:,2,1) = lnn(:,2,2)
    lnn(:,2,2) = lnn(:,2,3)
    lnTi(:,2,1) = lnTi(:,2,2)
    lnTi(:,2,2) = lnTi(:,2,3)
    lnTe(:,2,1) = lnTe(:,2,2)
    lnTe(:,2,2) = lnTe(:,2,3)
    vp(:,2,1) = vp(:,2,2)
    vp(:,2,2) = vp(:,2,3)
    w(:,2,1) = w(:,2,2)
    w(:,2,2) = w(:,2,3)
    psihat(:,2,1) = psihat(:,2,2)
    psihat(:,2,2) = psihat(:,2,3)

  end subroutine leapfrog


end module integrators
