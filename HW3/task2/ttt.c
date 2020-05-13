//mpicc task2.c -o task2.out
//mpirun -n 4 ./task2.out

#include"mpi.h"
#include<stdlib.h>
#include<stdbool.h>
#include<stdio.h>
#include<string.h>
#include<math.h>

typedef struct ca1d ca1d;

struct ca1d
{
    char* cells;
    char* next_state;
    int cellcount;
    unsigned char rule;
    int iterations;
    char rule_binary[9];

};

static void set_rule_binary(ca1d* ca);
static void calculate_next_state(ca1d* ca);
ca1d* ca1d_init(int cellcount_local, char* initialpattern_local, unsigned char rule, int iterations); 
void ca1d_free(ca1d* ca);


int main(int argc, char* argv[])
{
    int cellcount = 32;
    char initialpattern[cellcount];
    unsigned char rule = 18; //110 //90
    int iterations = 16;
    int rank;
    int procs;
    int previous_rank;
    int next_rank;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    int cellcount_local = cellcount/procs;
    char initialpattern_local[cellcount_local];

    char *recieve = (char *) malloc(sizeof(char)*cellcount_local);
    char result[iterations][cellcount];

    for (int i = 0; i < cellcount; i++) 
    {
        initialpattern[i] = '0';
    }
    initialpattern[cellcount/2+1] = '1';
    
    for (int n = rank*cellcount_local; n < (rank+1)*cellcount_local; n++) 
    {
        initialpattern_local[n-rank*cellcount_local] = initialpattern[n]; 
    }

    if (rank == 0)
    {
        puts("-------------------------");
        puts("Code based on");
        puts("https://www.codedrome.com/one-dimensional-cellular-automata-in-c/");
        printf("Using rule: %d\n", rule);
        printf("Number of iterations: %d\n", iterations);
        puts("-------------------------\n");  
        
        printf("initialpattern_local:        ");
        for (int i = 0; i < cellcount; i++) printf("%c", initialpattern[i]);
        printf("\n");  
  
        printf("initialpattern_local rank %d: ",rank);
        for (int i = 0; i < cellcount_local; i++) printf("%c", initialpattern_local[i]);
        printf("\n");          
    }
    else
    {
        printf("initialpattern_local rank %d: ",rank);
        for (int i = 0; i < cellcount_local; i++) printf("%c", initialpattern_local[i]);
        printf("\n");
    }
    
    
    ca1d* ca = ca1d_init(cellcount_local, initialpattern_local, rule, iterations);

    if (ca != NULL)
    {
        for(int i = 0; i < iterations; i++)
        {
            calculate_next_state(ca);

            if (rank == 0)
            {
                previous_rank = procs-1;
                next_rank = rank+1;
            }
            else if (rank == procs-1)
            {
                previous_rank = rank-1;
                next_rank = 0;
            }
            else
            {
                previous_rank = rank-1;
                next_rank = rank+1;
            }

            MPI_Request request[3];

            // send and receive next proc
            MPI_Isend(&ca->cells[cellcount_local], 1, MPI_CHAR, next_rank, 0, MPI_COMM_WORLD, &request[0]);
            MPI_Recv(&ca->cells[0], 1, MPI_CHAR, previous_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // send and recive previous proc
            MPI_Isend(&ca->cells[1], 1, MPI_CHAR, previous_rank, 1, MPI_COMM_WORLD, &request[1]);
            MPI_Recv(&ca->cells[cellcount_local+1], 1, MPI_CHAR, next_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (rank == 0) // gathering
            {
                for (int j = 1; j < cellcount_local+1; j++) //save info from proc 0
                {
                    result[i][j-1] = ca->cells[j]; 
                    //printf("%с\n", result);
                }                
                for (int proc = 1; proc < procs; proc++) // get and save info from other procs
                {  
                    MPI_Recv(recieve, cellcount_local, MPI_CHAR, proc, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
                    for (int j = 1; j < cellcount_local+1; j++) 
                    {
                        result[i][proc*cellcount_local+j-1] = recieve[j];  
                    }
                }
            }
            else //send info from proc != 0
            {
                MPI_Isend(&ca->cells, cellcount_local, MPI_CHAR, 0, 2, MPI_COMM_WORLD, &request[2]);
            }
            
        }
	    //printf("out of here\n");

        if (rank == 0)
        {
	        //printf("INSIDE\n");
            for (int iteration = 0; iteration < iterations+1; iteration++) 
            {
                for (int cell = 0; cell < ca->cellcount; cell++) 
                {
                    printf("|%-2d| %c\n", iteration, result[iteration][cell]);
                 }
            }
        }

	    MPI_Barrier(MPI_COMM_WORLD);
	    MPI_Finalize();
        free(recieve);
        ca1d_free(ca);
        return EXIT_SUCCESS;
    }
    else
    {
	    printf("FAIL");
	    MPI_Finalize();
        return EXIT_FAILURE;
    }
}

ca1d* ca1d_init(int cellcount_local, char* initialpattern_local, unsigned char rule, int iterations) 
{
    ca1d* ca = malloc(sizeof(ca1d));

    if(ca != NULL)
    {
         *ca = (ca1d){.cells = malloc(cellcount_local * sizeof(char)),
                    .next_state = malloc(cellcount_local * sizeof(char)),
                    .cellcount = cellcount_local,
                    .rule = rule,
                    .iterations = iterations,
                    .rule_binary = ""};


        if(ca->cells != NULL && ca->next_state != NULL)
        {
            set_rule_binary(ca);

            for(int i = 0; i < cellcount_local; i++)
            {
                if(initialpattern_local[i] == '0')
                {
                    ca->cells[i] = '0';
                }
                else if(initialpattern_local[i] == '1')
                {
                    ca->cells[i] = '1';
                }
            }


            return ca;
        }
        else
        {
            ca1d_free(ca);

            return NULL;
        }
    }
    else
    {
        return NULL;
    }
}

static void set_rule_binary(ca1d* ca)
{
    for(int p = 0; p <= 7; p++)
    {
        if((int)(pow(2, p)) & ca->rule)
        {
            ca->rule_binary[abs(p - 7)] = '1';
        }
        else
        {
            ca->rule_binary[abs(p - 7)] = '0';
        }
    }
    ca->rule_binary[8] = '\0';
}

static void calculate_next_state(ca1d* ca)
{
    int prev_index;
    int next_index;
    char neighbourhood[4];

    for(int i = 0; i < ca->cellcount; i++)
    {
        if (i == 0)
            // roll beginning round to end
            prev_index = ca->cellcount - 1;
        else
            prev_index = i - 1;

        if (i == (ca->cellcount - 1))
            // roll end round to beginning
            next_index = 0;
        else
            next_index = i + 1;

        // set neighbourhood of 3 cells
        neighbourhood[0] = ca->cells[prev_index];
        neighbourhood[1] = ca->cells[i];
        neighbourhood[2] = ca->cells[next_index];
        neighbourhood[3] = '\0';

        // set next cell state depending on neighbourhood
        if(strcmp(neighbourhood, "111") == 0)
            ca->next_state[i] = ca->rule_binary[0];
        else if(strcmp(neighbourhood, "110") == 0)
            ca->next_state[i] = ca->rule_binary[1];
        else if(strcmp(neighbourhood, "101") == 0)
            ca->next_state[i] = ca->rule_binary[2];
        else if(strcmp(neighbourhood, "100") == 0)
            ca->next_state[i] = ca->rule_binary[3];
        else if(strcmp(neighbourhood, "011") == 0)
            ca->next_state[i] = ca->rule_binary[4];
        else if(strcmp(neighbourhood, "010") == 0)
            ca->next_state[i] = ca->rule_binary[5];
        else if(strcmp(neighbourhood, "001") == 0)
            ca->next_state[i] = ca->rule_binary[6];
        else if(strcmp(neighbourhood, "000") == 0)
            ca->next_state[i] = ca->rule_binary[7];
    }

    // copy next state to current
    for (int i = 0; i < ca->cellcount; ca->cells[i] = ca->next_state[i], i++);
}

void ca1d_free(ca1d* ca)
{
    if(ca != NULL)
    {
        if(ca->cells != NULL)
            free(ca->cells);

        if(ca->next_state != NULL)
            free(ca->next_state);

        free(ca);
    }
}


