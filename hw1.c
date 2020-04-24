// gcc hw1.c -lm -pthread
// ./a.out

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <semaphore.h>
#include <sys/time.h>
#define BEGIN(rank,threads,n) ((rank)*(n)/(threads))
#define END(rank,threads,n)   (BEGIN((rank)+1,threads,n)-1)

double a = 0;                // left point
double b = 1;              // right point
int n = 100000000;                   // number of discretization points
//int n = 5;                   
double h;                 // distance between neighboring discretization points
int TOTAL_THREADS = 16;

pthread_mutex_t mutex;
sem_t semaphore;
                
int wait_flag = 0;

double busy_wait_res = 0;      // result of calculation using busy wait method
double mutex_res = 0;          // result of calculation using mutex method
double semaphore_res = 0;      // result of calculation using semaphore method

double serial_arc_length();

void* busy_wait_arc_length(void* rank);
void busy_wait_main();

void* mutex_arc_length(void*);
void mutex_main();

void* semaphore_arc_length(void*);
void semaphore_main();

double f(double x)
{
  return 4/(1+pow(x,2)); 
}

unsigned long get_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    unsigned long ret = tv.tv_usec;
    ret /= 1000;
    ret += (tv.tv_sec * 1000);
    return ret;
}

int main(int argc, char *argv[]) {
    
    h = (b-a)/n;

    printf("TOTAL NUMBER OF THREADS: %d\n", TOTAL_THREADS); 
    
    long start = get_time();
    double result = serial_arc_length();
    double duration = (get_time() - start);
    printf("solution on a single thread: %f, time: %f milliseconds\n", result, duration);
    
    busy_wait_main(); 
    mutex_main();
    semaphore_main();

    return 0;
}

double serial_arc_length()
{
    double sum = 0;
    double x;

    for (int i=0; i<=n; i++)
    {   
        x = (i+0.5)*h;
        sum += f(x);
    }
    return sum*h;
}

void* busy_wait_arc_length(void* rank)
{
    long thread_rank = (long)rank;
    double sum = 0;
    double x;
 
    long begin = BEGIN(thread_rank, TOTAL_THREADS, n);
    long end = END(thread_rank, TOTAL_THREADS, n);

    for (int i=begin; i<=end; i++)
    {   
        x = (i+0.5)*h;
        sum += f(x);
    }
    sum = sum*h;

    while (wait_flag != thread_rank){};
    busy_wait_res += sum;
    wait_flag++;

    return NULL;
}

void busy_wait_main()
{
    pthread_t* thread_ptr;
    thread_ptr = malloc(TOTAL_THREADS * sizeof(pthread_t));

    long start = get_time();
    double duration;

    for (long i = 0; i < TOTAL_THREADS; i++)
    {
       pthread_create(&thread_ptr[i], NULL, busy_wait_arc_length, (void*)i);
    }

    for (long i = 0; i < TOTAL_THREADS; i++)
    {
       pthread_join(thread_ptr[i], NULL);
    }

    duration = (get_time() - start);
    printf("solution using busy waiting: %f, time: %f milliseconds\n", busy_wait_res, duration);

    free(thread_ptr);    
}


void* mutex_arc_length(void* rank)
{
    long thread_rank = (long)rank;
    double sum = 0;
    double x;
 
    long begin = BEGIN(thread_rank, TOTAL_THREADS, n);
    long end = END(thread_rank, TOTAL_THREADS, n);

    for (int i=begin; i<=end; i++)
    {   
        x = (i+0.5)*h;
        sum += f(x);
    }
    sum = sum*h;

    pthread_mutex_lock(&mutex);
    mutex_res += sum;
    pthread_mutex_unlock(&mutex);
    
    return NULL;

}

void mutex_main()
{
    pthread_t* thread_ptr;
    pthread_mutex_init(&mutex, NULL);
    thread_ptr = malloc(TOTAL_THREADS * sizeof(pthread_t));

    long start = get_time();
    double duration;

    for (long i = 0; i < TOTAL_THREADS; i++)
    {
       pthread_create(&thread_ptr[i], NULL, mutex_arc_length, (void*)i);
    }

    for (long i = 0; i < TOTAL_THREADS; i++)
    {
       pthread_join(thread_ptr[i], NULL);
    }

    duration = (get_time() - start);
    printf("solution using mutex: %f, time: %f milliseconds\n", mutex_res, duration);

    free(thread_ptr);
    pthread_mutex_destroy(&mutex);    
}


void* semaphore_arc_length(void* rank)
{
    long thread_rank = (long)rank;
    double sum = 0;
    double x;
 
    long begin = BEGIN(thread_rank, TOTAL_THREADS, n);
    long end = END(thread_rank, TOTAL_THREADS, n);

    for (int i=begin; i<=end; i++)
    {   
        x = (i+0.5)*h;
        sum += f(x);
    }
    sum = sum*h;

    sem_wait(&semaphore);
    semaphore_res += sum;
    sem_post(&semaphore);
    
    return NULL;
}

void semaphore_main()
{
    pthread_t* thread_ptr;
    sem_init(&semaphore, 0, 1);
    thread_ptr = malloc(TOTAL_THREADS * sizeof(pthread_t));

    long start = get_time();
    double duration;

    for (long i = 0; i < TOTAL_THREADS; i++)
    {
       pthread_create(&thread_ptr[i], NULL, semaphore_arc_length, (void*)i);
    }

    for (long i = 0; i < TOTAL_THREADS; i++)
    {
       pthread_join(thread_ptr[i], NULL);
    }

    duration = (get_time() - start);
    printf("solution using semaphore: %f, time: %f milliseconds\n", semaphore_res, duration);

    free(thread_ptr);
    sem_destroy(&semaphore);    
}
