//mpicc PingPong.c -o PingPong.out
//mpirun -n 3 ./PingPong.out

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <sys/time.h>
#define N 1000000

int main(int argc, char *argv[])
{
    int next, previous;
    int rank;
    int procs;
    int message[N]; 
    double start_time, start_full_time;

    printf("N = %d \n", N);

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    
    for (int i = 0; i < N; i++) message[i] = i; 
    
    if (procs != 3) 
    { 
        printf("In this game 3 Persons are playing, write: \n mpicc PingPong.c -o PingPong.out \n mpirun -n 3 ./PingPong.out\n");
    }  

    else
    {  
        if (rank == 0)   
        {  

            next = rank+1;
            
            //printf("Person %d sending message to Person %d \n", rank, next); 

            start_full_time = MPI_Wtime();

            MPI_Send(message, N, MPI_INT, next, 0, MPI_COMM_WORLD);

            previous = procs-1;

            MPI_Recv(message, N, MPI_INT, previous, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            //printf("Person %d got message back from Person %d \n", rank, previous);

            printf("%g seconds between Person 0 send and receive message (Total time)\n ", MPI_Wtime() - start_full_time);

        } 

        if (rank == 1)
        {
            //printf("Person %d waiting message\n", rank);
            
            previous =  rank-1;
            
            start_time = MPI_Wtime();

            MPI_Recv(message, N, MPI_INT, previous, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            next = rank+1;
           
            //printf("Name from Person %d came to Person %d, sending message to Person %d \n", previous, rank, next);

            MPI_Send(message, N, MPI_INT, next, 0, MPI_COMM_WORLD);
            
            printf("%g seconds between Person 1 send and receive message (time per Person)\n ", MPI_Wtime() - start_time);

        }  

        if (rank == 2)
        {
            //printf("Person %d waiting message\n", rank);
            
            previous =  rank-1;

            MPI_Recv(message, N, MPI_INT, previous, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            next = 0;
           
            //printf("Message from person %d came to person %d, sending message to person %d \n", previous, rank, next);

            MPI_Send(message, N, MPI_INT, next, 0, MPI_COMM_WORLD);
        } 
       
           
    }
    
    MPI_Finalize();

    exit(0);

}
