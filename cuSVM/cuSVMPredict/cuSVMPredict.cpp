 
#include <stdio.h> 
#include <math.h>


#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "svm.h"



extern "C"
void GPUPredictWrapper (int m, int n, int k, float kernelwidth, const float *Test, const float *Svs, float * alphas,float *prediction, float beta,float isregression);



struct svm_node *x;
int max_nr_attr = 64;

struct svm_model* model;
int predict_probability=0;



void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

void predict(FILE *input, FILE *output)
{
	int correct = 0;
	int total = 0;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

	int svm_type=svm_get_svm_type(model);
	int nr_class=svm_get_nr_class(model);
	double *prob_estimates=NULL;
	int j;

	if(predict_probability)
	{
		if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
			printf("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g\n",svm_get_svr_probability(model));
		else
		{
			int *labels=(int *) malloc(nr_class*sizeof(int));
			svm_get_labels(model,labels);
			prob_estimates = (double *) malloc(nr_class*sizeof(double));
			fprintf(output,"labels");		
			for(j=0;j<nr_class;j++)
				fprintf(output," %d",labels[j]);
			fprintf(output,"\n");
			free(labels);
		}
	}

	max_line_len = 1024;
	line = (char *)malloc(max_line_len*sizeof(char));
	while(readline(input) != NULL)
	{
		int i = 0;
		double target_label, predict_label;
		char *idx, *val, *label, *endptr;
		int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0

		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(total+1);

		target_label = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(total+1);

		while(1)
		{
			if(i>=max_nr_attr-1)	// need one more for index = -1
			{
				max_nr_attr *= 2;
				x = (struct svm_node *) realloc(x,max_nr_attr*sizeof(struct svm_node));
			}

			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;
			errno = 0;
			x[i].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
				exit_input_error(total+1);
			else
				inst_max_index = x[i].index;

			errno = 0;
			x[i].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(total+1);

			++i;
		}
		x[i].index = -1;

		{
		//	predict_label = svm_predict(model,x);
			fprintf(output,"%g\n",predict_label);
		}


		
		if(predict_label == target_label)
			++correct;
		error += (predict_label-target_label)*(predict_label-target_label);
		sump += predict_label;
		sumt += target_label;
		sumpp += predict_label*predict_label;
		sumtt += target_label*target_label;
		sumpt += predict_label*target_label;
		++total;
	}
		printf("Accuracy = %g%% (%d/%d) (classification)\n",
		       (double)correct/total*100,correct,total);
	if(predict_probability)
		free(prob_estimates);
}

void exit_with_help()
{
	printf(
	"Usage: svm-predict [options] test_file model_file output_file\n"
	"options:\n"
	"-b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); for one-class SVM only 0 is supported\n"
	);
	exit(1);
}



int main(int argc, char **argv)
{
	FILE *input, *output;
	int i;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;
		switch(argv[i-1][1])
		{
			case 'b':
				predict_probability = atoi(argv[i]);
				break;
			default:
				fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
		}
	}
	if(i>=argc-2)
		exit_with_help();
	
	input = fopen(argv[i],"r");
	if(input == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",argv[i]);
		exit(1);
	}

	output = fopen(argv[i+2],"w");
	if(output == NULL)
	{
		fprintf(stderr,"can't open output file %s\n",argv[i+2]);
		exit(1);
	}

	if((model=svm_load_model(argv[i+1]))==0)
	{
		fprintf(stderr,"can't open model file %s\n",argv[i+1]);
		exit(1);
	}

	x = (struct svm_node *) malloc(max_nr_attr*sizeof(struct svm_node));

	// model is loaded now
	/*
	float *input;
	float *supportVectors;
	float *alphas;
	float *predictions;
	float *bias;
	bool isregression = 0;
	float *kernelwidth = model.gamma;
	int m = 0;
	int n = 0;
	int k = 0;
	GPUPredictWrapper(m, n, k, kernelwidth, input, supportVectors, alphas, predictions, bias, isregression);
	
	predict(input,output);

	svm_free_and_destroy_model(&model);
	free(x);
	free(line);
	fclose(input);
	fclose(output);
	return 0;
	*/
}



/*

// all single precision floats.
Test data, SupportVectors, alpha, beta=bias, gamma, isRegression
{
	
	int m = mxGetM(prhs[0]);
	int k = mxGetN(prhs[0]);
	
	//B is transposed in the RBFKernel gpu function
	int n=mxGetM(prhs[1]);
	int testk=mxGetN(prhs[1]);

// some checks	

	float* Test=(float *)mxGetData(prhs[0]);
	float* Svs=(float *)mxGetData(prhs[1]);
	float* alphas=(float *)mxGetData(prhs[2]);
	float beta=*(float *)mxGetData(prhs[3]);

	float kernelwidth;
	float isregression;
	
	// create label matrix, with floats and length m
	float* prediction = new[];

	GPUPredictWrapper(m, n, k, kernelwidth, Test, Svs, alphas, prediction, beta, isregression);
	return;
}


*/