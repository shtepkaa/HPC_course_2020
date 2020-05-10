//mpicc PingPong.c -o PingPong.out
//mpirun -n 3 ./PingPong.out

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

int main(int argc, char *argv[])
{
    int next, previous;
    int rank;
    int procs;
    int names[procs]; 
    double start_time, Time;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
 
    if (procs != 3) 
    { 
        printf("In this game 3 persons are playing, write: \n mpicc PingPong.c -o PingPong.out \n mpirun -n 3 ./PingPong.out\n");
    }  
    
    else   
    {

        if (rank == 0)   
        {  
            names[rank] = rank;

            next = rank+1;
            
            printf("Person %d sending name to person %d \n", rank, next); 
            
            MPI_Send(names, procs, MPI_INT, next, 0, MPI_COMM_WORLD);

            previous = procs-1;

            MPI_Recv(names, procs, MPI_INT, previous, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            printf("Person %d got names back from person %d \n", rank, previous);

        } 

        if (rank == 1)
        {
            printf("Person %d waiting name\n", rank);
            
            previous =  rank-1;

            MPI_Recv(names, procs, MPI_INT, previous, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            names[rank] = rank;
            
            next = rank+1;
           
            printf("Name from person %d came to person %d, sending himself name to person %d \n", previous, rank, next);

            MPI_Send(names, procs, MPI_INT, next, 0, MPI_COMM_WORLD);
        }  

        if (rank == 2)
        {
            printf("Person %d waiting name\n", rank);
            
            previous =  rank-1;

            MPI_Recv(names, procs, MPI_INT, previous, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            names[rank] = rank;
            
            next = 0;
           
            printf("Name from person %d came to person %d, sending himself name to person %d \n", previous, rank, next);

            MPI_Send(names, procs, MPI_INT, next, 0, MPI_COMM_WORLD);
        } 
            

    }

    MPI_Finalize();

    exit(0);

}
