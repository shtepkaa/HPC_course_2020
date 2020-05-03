#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <omp.h>

void array_filled(float **array, const int n) 
{
	*array = (float *) malloc(n*sizeof(float));

	for (int i = 0; i < n*n; i++) 
    {
		(*array)[i] = ((float)rand()/(float)(RAND_MAX))*100;
	}
}

int main(int argc, char **argv) 
{

  float *x, *y;
  double SUMx, SUMy, SUMxy, SUMxx, SUMres, res, slope, y_intercept, y_estimate ;
  int i, n;
  
  const int threads = omp_get_max_threads();

  n = 10;
  array_filled(&x,n);
  array_filled(&y,n);

  SUMx = 0; SUMy = 0; SUMxy = 0; SUMxx = 0;

  #pragma omp parallel for private(i) shared(slope, y_intercept) num_threads(threads) reduction(+:SUMx, SUMy, SUMxy, SUMxx)
  
  for (i = 0; i < n; i++) 
  {
        SUMx += x[i];
        SUMy += y[i];
        SUMxy += x[i]*y[i];
        SUMxx += x[i]*x[i];
  }

  slope = (SUMx*SUMy - n*SUMxy) / (SUMx*SUMx - n*SUMxx);
  y_intercept = (SUMy - slope*SUMx) / n;

  printf ("\n");
  printf ("The linear equation that best fits the given data:\n");
  printf ("       y = %6.2lfx + %6.2lf\n", slope, y_intercept);
  printf ("--------------------------------------------------\n");
  printf ("   Original (x,y)     Estimated y     Residual\n");
  printf ("--------------------------------------------------\n");


  SUMres = 0;

  for (i=0; i<n; i++) 
  {
        y_estimate = slope*x[i] + y_intercept;
        res = y[i] - y_estimate;
        SUMres += res*res;
        printf ("   (%6.2lf %6.2lf)      %6.2lf       %6.2lf\n", x[i], y[i], y_estimate, res);
  }
  printf("--------------------------------------------------\n");
  printf("Residual sum = %6.2lf\n", SUMres);
  
  return 0;

}

