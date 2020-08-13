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

!!  Umbrella Intetration derivates of igrid  !!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine derivate_igrid(grid, a_fc, a_rc0, a_mean, a_prec, a_det_sqrt, a_N, beta, impossible, n, n_ig, dA_grid)

  !--------------------------------------------------------------------
  ! Calculate Umbrella Integration's derivates in a incomplete grid
  !--------------------------------------------------------------------

  implicit none

  ! Variable definition
  integer, intent(in)  :: n
  integer, intent(in)  :: n_ig
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
  real(8), parameter   :: thr = 1.D-9     ! consider impossible if lower
  real(8)              :: rc(2), normal, derivate(n,2)

  ! calculate gradient field of free energy over array grid [Kästner 2009 - Eq.11]
  do ig=1,n_ig
    rc = grid(ig,:)
    normal = normal_tot(rc)
    if (normal > thr) then
      ! calculate derivate term of every data point on that grid point
      do i=1,n
        derivate(i,:) = a_N(i)*probability(rc,i)/normal * dA(rc,i)
      enddo
      ! sum all the derivates on that grid point
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
                    / (a_det_sqrt(i) * 8.D0*DATAN(1.D0))

    end function

    ! local derivate of free energy [Kästner 2009 - Eq.10] ------------
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
