!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!         Umbrella Integrator - Fortranized Key Subroutines         !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! f2py compilation example:
! python3 -m numpy.f2py -c umbrellaint_fortran.f90 -m umbrellaint_fortran --opt='-Ofast'

!!  point_in  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine point_in(coord, grid, thr, result, n)

  !--------------------------------------------------------------------
  ! Evaluate if a point is close to any grid point
  !--------------------------------------------------------------------

  implicit none

  ! Variable definition
  integer, intent(in)  :: n
  real(8), intent(in)  :: coord(2), grid(n,2), thr

  logical, intent(out) :: result

  integer              :: i
  real(8)              :: grid_dist(n)


  ! distance of grid to point
  do i=1,n
    grid_dist(i) = (grid(i,1)-coord(1))**2 + (grid(i,2)-coord(2))**2
  enddo

  ! result true if less than thr, else false
  result = minval(grid_dist) <= thr**2

end subroutine

!!  Umbrella Integration derivatives in 1D  !!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine ui_derivate_1d(bins, a_fc, a_rc0, a_mean, a_std, a_N, beta, n, n_bins, dA_bins)

  !--------------------------------------------------------------------
  ! Calculate Umbrella Integration's derivatives in 1D
  !--------------------------------------------------------------------

  implicit none

  ! Variable definition
  integer, intent(in)  :: n                   ! No data elements
  integer, intent(in)  :: n_bins              ! No bins elements
  real(8), intent(in)  :: bins(n_bins)
  real(8), intent(in)  :: a_fc(n)
  real(8), intent(in)  :: a_rc0(n)
  real(8), intent(in)  :: a_mean(n)
  real(8), intent(in)  :: a_std(n)
  real(8), intent(in)  :: a_N(n)
  real(8), intent(in)  :: beta

  real(8), intent(out) :: dA_bins(n_bins)

  integer              :: ib, i
  real(8)              :: rc, normal, derivate(n)
  real(8), parameter   :: TAU_SQRT=SQRT(8.D0*DATAN(1.D0))

  ! calculate gradient field of free energy over array grid [Kästner 2005 - Eq.7]
  do ib=1,n_bins
    rc = bins(ib)
    normal = normal_tot(rc)
    ! calculate derivative term of every data point on that grid point
    do i=1,n
      derivate(i) = a_N(i)*probability(rc,i)/normal * dA(rc,i)
    enddo
    ! sum all the derivatives on that grid point
    dA_bins(ib) = SUM(derivate)
  enddo


  contains

    ! normal probability [Kästner 2005 - Eq.5] ------------------------
    function probability(rc, i)

      implicit none

      real(8), intent(in)  :: rc
      integer, intent(in)  :: i

      real(8)              :: probability

      real(8)              :: diff

      diff = rc - a_mean(i)
      probability = EXP(-0.5D0 * (diff/a_std(i))**2) / (a_std(i)*TAU_SQRT)

    end function

    ! local derivative of free energy [Kästner 2005 - Eq.6] ------------
    function dA(rc, i)

      implicit none

      real(8), intent(in)  :: rc
      integer, intent(in)  :: i

      real(8)              :: dA

      dA = (rc - a_mean(i)) / (beta * a_std(i)**2) - a_fc(i) * (rc - a_rc0(i))

    end function

    ! normalization total [Kästner 2005 - Eq.8] ----------------------
    function normal_tot(rc)

      implicit none

      real(8), intent(in)  :: rc

      real(8)              :: normal_tot

      integer              :: i

      normal_tot = 0.D0
      do i=1,n
        normal_tot = normal_tot + a_N(i) * probability(rc, i)
      enddo

    end function

end subroutine

!!  Umbrella Integration derivatives in 2D  !!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine ui_derivate_2d_rgrid(grid, m_fc, m_rc0, m_mean, m_prec, m_det_sqrt, m_N, beta, ni, nj, n_ig, n_jg, dA_grid)

  !--------------------------------------------------------------------
  ! Calculate Umbrella Integration's derivatives in a rectangular grid
  !--------------------------------------------------------------------

  implicit none

  ! Variable definition
  integer, intent(in)  :: ni, nj              ! Dimension of data
  integer, intent(in)  :: n_ig, n_jg          ! Dimension of grid
  real(8), intent(in)  :: grid(n_jg,n_ig,2)
  real(8), intent(in)  :: m_fc(nj,ni,2,2)
  real(8), intent(in)  :: m_rc0(nj,ni,2)
  real(8), intent(in)  :: m_mean(nj,ni,2)
  real(8), intent(in)  :: m_prec(nj,ni,2,2)
  real(8), intent(in)  :: m_det_sqrt(nj,ni)
  real(8), intent(in)  :: m_N(nj,ni)
  real(8), intent(in)  :: beta

  real(8), intent(out) :: dA_grid(n_jg,n_ig,2)

  integer              :: ig, jg, i, j
  real(8)              :: rc(2), normal, derivate(nj,ni,2)
  real(8), parameter   :: TAU=8.D0*DATAN(1.D0)

  ! calculate gradient field of free energy over array grid [Kästner 2009 - Eq.11]
  do ig=1,n_ig
    do jg=1,n_jg
      rc = grid(jg,ig,:)
      normal = normal_tot(rc)
      ! calculate derivative term of every data point on that grid point
      do i=1,ni
        do j=1,nj
          derivate(j,i,:) = m_N(j,i)*probability(rc,j,i)/normal * dA(rc,j,i)
        enddo
      enddo
      ! sum all the derivatives on that grid point
      dA_grid(jg,ig,:) = (/ 0.D0, 0.D0 /)
      do i=1,ni
        do j=1,nj
          dA_grid(jg,ig,:) = dA_grid(jg,ig,:) + derivate(j,i,:)
        enddo
      enddo
    enddo
  enddo


  contains

    ! normal probability [Kästner 2009 - Eq.9] ------------------------
    function probability(rc, j, i)

      implicit none

      real(8), intent(in)  :: rc(2)
      integer, intent(in)  :: j, i

      real(8)              :: probability

      real(8)              :: diff(2)

      diff = rc - m_mean(j,i,:)
      probability = EXP( -0.5D0 * DOT_PRODUCT(diff,MATMUL(m_prec(j,i,:,:),diff)) )  &
                    / (m_det_sqrt(j,i) * TAU)

    end function

    ! local derivative of free energy [Kästner 2009 - Eq.10] ------------
    function dA(rc, j, i)

      implicit none

      real(8), intent(in)  :: rc(2)
      integer, intent(in)  :: j, i

      real(8)              :: dA(2)

      dA = MATMUL((rc - m_mean(j,i,:))/beta, m_prec(j,i,:,:))  &
           - MATMUL((rc - m_rc0(j,i,:)), m_fc(j,i,:,:))

    end function

    ! normalization total [Kästner 2009 - Eq.11] ----------------------
    function normal_tot(rc)

      implicit none

      real(8), intent(in)  :: rc(2)

      real(8)              :: normal_tot

      integer              :: j, i

      normal_tot = 0.D0
      do i=1,ni
        do j=1,nj
          normal_tot = normal_tot + m_N(j,i) * probability(rc, j, i)
        enddo
      enddo

    end function

end subroutine

!!  Umbrella Integration derivatives for igrid in 2D  !!!!!!!!!!!!!!!!!
subroutine ui_derivate_2d_igrid(grid, a_fc, a_rc0, a_mean, a_prec, a_det_sqrt, a_N, beta, impossible, n, n_ig, dA_grid)

  !--------------------------------------------------------------------
  ! Calculate Umbrella Integration's derivatives in a incomplete grid
  !--------------------------------------------------------------------

  implicit none

  ! Variable definition
  integer, intent(in)  :: n                   ! No data elements
  integer, intent(in)  :: n_ig                ! No grid elements
  real(8), intent(in)  :: grid(n_ig,2)
  real(8), intent(in)  :: a_fc(n,2,2)
  real(8), intent(in)  :: a_rc0(n,2)
  real(8), intent(in)  :: a_mean(n,2)
  real(8), intent(in)  :: a_prec(n,2,2)
  real(8), intent(in)  :: a_det_sqrt(n)
  real(8), intent(in)  :: a_N(n)
  real(8), intent(in)  :: beta
  real(8), intent(in)  :: impossible

  real(8), intent(out) :: dA_grid(n_ig,2)

  integer              :: ig, i
  real(8)              :: rc(2), normal, derivate(n,2)
  real(8), parameter   :: thr=1.D-9           ! consider impossible if lower
  real(8), parameter   :: TAU=8.D0*DATAN(1.D0)

  ! calculate gradient field of free energy over array grid [Kästner 2009 - Eq.11]
  do ig=1,n_ig
    rc = grid(ig,:)
    normal = normal_tot(rc)
    if (normal > thr) then
      ! calculate derivative term of every data point on that grid point
      do i=1,n
        derivate(i,:) = a_N(i)*probability(rc,i)/normal * dA(rc,i)
      enddo
      ! sum all the derivatives on that grid point
      dA_grid(ig,:) = (/ 0.D0, 0.D0 /)
      do i=1,n
        dA_grid(ig,:) = dA_grid(ig,:) + derivate(i,:)
      enddo
    else
      dA_grid(ig,:) = (/ impossible, impossible /)
    endif
  enddo


  contains

    ! normal probability [Kästner 2009 - Eq.9] ------------------------
    function probability(rc, i)

      implicit none

      real(8), intent(in)  :: rc(2)
      integer, intent(in)  :: i

      real(8)              :: probability

      real(8)              :: diff(2)

      diff = rc - a_mean(i,:)
      probability = EXP( -0.5D0 * DOT_PRODUCT(diff,MATMUL(a_prec(i,:,:),diff)) )  &
                    / (a_det_sqrt(i) * TAU)

    end function

    ! local derivative of free energy [Kästner 2009 - Eq.10] ------------
    function dA(rc, i)

      implicit none

      real(8), intent(in)  :: rc(2)
      integer, intent(in)  :: i

      real(8)              :: dA(2)

      dA = MATMUL((rc - a_mean(i,:))/beta, a_prec(i,:,:))  &
           - MATMUL((rc - a_rc0(i,:)), a_fc(i,:,:))

    end function

    ! normalization total [Kästner 2009 - Eq.11] ----------------------
    function normal_tot(rc)

      implicit none

      real(8), intent(in)  :: rc(2)

      real(8)              :: normal_tot

      integer              :: i

      normal_tot = 0.D0
      do i=1,n
        normal_tot = normal_tot + a_N(i) * probability(rc, i)
      enddo

    end function

end subroutine
