
# set compiler
FC = mpif90
MODS = pde.mod comms.mod model.mod integrators.mod
H5LIB = usr/lib/x86_64-linux-gnu/hdf5/serial
H5INC = usr/include/hdf5/serial

# make executable
gdb.exe: main.o
	$(FC) -L/$(H5LIB) main.o -o gdb.exe -lhdf5 -lhdf5_fortran

# compile program
main.o: main.f90 $(MODS)
	$(FC) -c main.f90 -I/$(H5INC)

# compile modules
$(MODS): gdbmods.f90
	$(FC) -c gdbmods.f90 -I/$(H5INC)



clean:
	rm -f *.mod *.o gdb.exe
