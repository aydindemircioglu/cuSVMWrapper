//===========================================================================
/*!
 *
 * \brief       cuSVM wrapper.
 *
 * \author      Aydin Demircioglu
 * \date        2015
 *
 *
 * This is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You have not received a copy of the GNU Lesser General Public License
 * along with this. See <http://www.gnu.org/licenses/>.
 *
*/
//===========================================================================


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include "svm.h"
#include <math.h>
#include <string>
#include <iostream>


extern "C" {
	void SVMTrain(float *mexalpha,float* beta,float*y,float *x ,float CC, float gamma, int m, int n, float StoppingCrit);
	void SVRTrain(float *mexalpha,float* beta,float*y,float *x ,float CC, float gamma, float eps, int m, int n, float StoppingCrit);
}

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define Calloc(type,n) (type *)calloc((n), sizeof(type))

void print_null(const char *s) {}

void exit_with_help()
{
	printf(
		"Usage: svm-train [options] training_set_file [model_file]\n"
		"options:\n"
		"-s svm_type : set type of SVM (default 0)\n"
		"	0 -- C-SVC		(multi-class classification)\n"
//		"	3 -- epsilon-SVR	(regression)\n"
		"-t kernel_type : set type of kernel function (default 2)\n"
		"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
		"-g gamma : set gamma in kernel function (default 1/num_features)\n"
		"-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
//		"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
		"-m cachesize : set cache memory size in MB (default 100)\n"
		"-e epsilon : set tolerance of termination criterion (default 0.001)\n"
//		"-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
		"-q : quiet mode (no outputs)\n"
	);
	exit(1);
}



void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}



// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
static void svm_group_classes(const svm_problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
	int l = prob->l;
	int max_nr_class = 16;
	int nr_class = 0;
	int *label = Malloc(int,max_nr_class);
	int *count = Malloc(int,max_nr_class);
	int *data_label = Malloc(int,l);	
	int i;
	
	for(i=0;i<l;i++)
	{
		int this_label = (int)prob->y[i];
		int j;
		for(j=0;j<nr_class;j++)
		{
			if(this_label == label[j])
			{
				++count[j];
				break;
			}
		}
		data_label[i] = j;
		if(j == nr_class)
		{
			if(nr_class == max_nr_class)
			{
				max_nr_class *= 2;
				label = (int *)realloc(label,max_nr_class*sizeof(int));
				count = (int *)realloc(count,max_nr_class*sizeof(int));
			}
			label[nr_class] = this_label;
			count[nr_class] = 1;
			++nr_class;
		}
	}
	
	//
	// Labels are ordered by their first occurrence in the training set. 
	// However, for two-class sets with -1/+1 labels and -1 appears first, 
	// we swap labels to ensure that internally the binary SVM has positive data corresponding to the +1 instances.
	//
	if (nr_class == 2 && label[0] == -1 && label[1] == 1)
	{
		swap(label[0],label[1]);
		swap(count[0],count[1]);
		for(i=0;i<l;i++)
		{
			if(data_label[i] == 0)
				data_label[i] = 1;
			else
				data_label[i] = 0;
		}
	}
	
	int *start = Malloc(int,nr_class);
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[data_label[i]]] = i;
		++start[data_label[i]];
	}
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];
	
	*nr_class_ret = nr_class;
	*label_ret = label;
	*start_ret = start;
	*count_ret = count;
	free(data_label);
}




void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
void read_problem(const char *filename);
void do_cross_validation();

struct svm_parameter param;		// set by parse_command_line
struct svm_problem prob;		// set by read_problem
struct svm_model *model;
struct svm_node *x_space;
int cross_validation;
int nr_fold;

static char *line = NULL;
static int max_line_len;

#define DEBUG if (1 == 1) 

static char* readline(FILE *input)
{
	int len;
	
	if(fgets(line,max_line_len,input) == NULL)
		return NULL;
	
	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}


//extern double subsamplingAmount;

int main(int argc, char **argv)
{
	char input_file_name[1024];
	char model_file_name[1024];
	const char *error_msg;
	
	// parse CL and read problem.
	parse_command_line(argc, argv, input_file_name, model_file_name);
	read_problem(input_file_name);
	error_msg = svm_check_parameter(&prob,&param);
	
	if(error_msg) {
		fprintf(stderr,"ERROR: %s\n",error_msg);
		exit(1);
	}
	
	DEBUG std::cout << "Loaded training file with " << prob.l << " points and " << prob.max_index << " features.\n";
		
	//create the structures directly
	float *y = Calloc (float, prob.l); 
	// the index of a sparse file starts with 1:... -- at least we assume that here!!
	float *x = Calloc (float, prob.l * (prob.max_index) );
	
	for(int i = 0; i < prob.l; i++)
	{
		// elementwise copy over
		svm_node *px = prob.x[i];
		
		while(px->index != -1)
		{
			// the index of a sparse file starts with 1:... -- at least we assume that here.
			if (px -> index == 0) {
				throw "Features in sparse file must start with 1:.. but found 0:...!\n";
			}

			x[i*(prob.max_index) + px->index-1] = px -> value;
			++px;
		}
		
		// assign label
		y[i] = prob.y[i];
	}

	
	float C = param.C;
	float kernelwidth = param.gamma;
	float eps = param.eps;
	float bias = 0;
	
		
	// do extra check if there is only one class
	// do check for +1 and -1 labels, not 0/1
	int JustOneClassError=1;
	int NotOneorNegOneError=0;
	float FirstY=y[0];

	for(int k = 0; k < prob.l; k++)
	{
		if(y[k]!=FirstY) {
			JustOneClassError = 0;
		}
		
		if((y[k]!=1.0) && (y[k]!=-1.0) ) {
			NotOneorNegOneError=1;
		}
	}
	
	if (JustOneClassError==1)
		throw ("All training labels are of the same class.  There must of course be two classes");
	
	if (NotOneorNegOneError==1)
		throw ("Training labels must be either 1 or -1.");

	DEBUG std::cout << "Getting problem properties.\n";
	int l = prob.l;
	int nr_class = 2;
	int *label = NULL;
	int *start = NULL;
	int *count = NULL;
	int *perm = Malloc(int,l);
	
	// group training data of the same class
	svm_group_classes (&prob, &nr_class, &label, &start, &count, perm);
	DEBUG std::cout << "Found " << nr_class << " classes.\n";
	
/*	
	DEBUG 	for (int i = 0; i < prob.l * prob.max_index; i++) {
		if (i % prob.max_index == 0) std::cout << "\n" << y[i / prob.max_index] << " ";
		if (x[i] == 0) continue;
		std::cout << 1 + i%prob.max_index << ":" << x[i] << " ";
	}
	std::cout << "\n";*/
	
	// do the training now.
	float *alpha=Calloc(float, prob.l);
	
	// FIXME: KERNEL CACHE SIZE
	int IsRegression = 0;

	std::cout << "Starting Training on GPU\n";
	DEBUG std::cout << "Training with C: " << C<< "\n";
	DEBUG std::cout << "Training with gamma: " << kernelwidth << "\n";
	DEBUG std::cout << "Training with epsilon: " << eps << "\n";
	
	SVMTrain(alpha, &bias, y, x ,C, kernelwidth, prob.l, prob.max_index, eps); // m = prob.l n = prob.max_index
	
	std::cout << "Finished Training on GPU\n";
	DEBUG std::cout << "Bias: " <<  bias <<  "\n";
	
	// save  damn model
	struct svm_model* hack_model = Malloc(svm_model, 1);
	hack_model->param = param;

	double* coeff_ptr = Calloc(double, prob.l);
	hack_model->sv_coef = &coeff_ptr;
	hack_model->SV = Calloc(svm_node*, prob.l);
	int nonzero = 0;
	int positive = 0;
	int negative = 0;
	
	DEBUG std::cout << "Converting model.\n";
	SVC_Q Q(prob, param, y);
	for (int i=0; i < prob.l; i++)
	{
		if (alpha[i] != 0.0)
		{
			if (y[i]==+1)
			{
				coeff_ptr[nonzero] = alpha[i];
				positive++;
			}
			else
			{
				coeff_ptr[nonzero] = -alpha[i];
				negative++;
			}
			hack_model->SV[nonzero] = const_cast<struct svm_node *>(Q.get_x(i));
			nonzero++;
		}
	}
	DEBUG std::cout << "Found " << nonzero << " support vectors, " << positive << " positive and " << negative << " negative.\n";
	
	// FIXME: assume binary problem here!
	
	hack_model->rho = Malloc(double, 1);
	hack_model->nSV = Malloc(int, 2);
	hack_model->l = nonzero;
	hack_model->nSV[0] = positive;
	hack_model->nSV[1] = negative;
	hack_model->rho[0] = -bias;
	
	hack_model->nr_class = nr_class;
	
	hack_model->label = Malloc(int, nr_class);
	for(int j = 0; j < nr_class; j++)
		hack_model->label[j] = label[j];
	
	hack_model->probA=NULL;
	hack_model->probB=NULL;
	
	// write model to disk
	DEBUG std::cout << "Writing model to " << model_file_name << "\n";
	if(svm_save_model(model_file_name, hack_model))
	{
		fprintf(stderr, "can't save model to file %s\n", model_file_name);
		exit(1);
	}

	free(coeff_ptr);
	free(hack_model->SV);
	free (alpha);

	// yeah, we have a memory leak here, luckily we quit anyway.
	svm_destroy_param(&param);
	free(x);
	free(y);
	free(perm);
 	free(prob.y);
 	free(prob.x);
 	free(x_space);
 	free(line);
	return 0;
}



void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
	int i;
	void (*print_func)(const char*) = NULL;	// default printing to stdout
	
	// default values
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 0;	// 1/num_features
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 1;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	param.weight_label = Malloc (int, 2);
	param.weight = Malloc(double, 2);

	cross_validation = 0;
	
	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 's':
				param.svm_type = atoi(argv[i]);
				break;
			case 't':
				param.kernel_type = atoi(argv[i]);
				break;
			case 'd':
				param.degree = atoi(argv[i]);
				break;
			case 'g':
				param.gamma = atof(argv[i]);
				break;
			case 'r':
				param.coef0 = atof(argv[i]);
				break;
			case 'n':
				param.nu = atof(argv[i]);
				break;
			case 'm':
				param.cache_size = atof(argv[i]);
				break;
			case 'c':
				param.C = atof(argv[i]);
				break;
			case 'e':
				param.eps = atof(argv[i]);
				break;
			case 'p':
				param.p = atof(argv[i]);
				break;
			case 'h':
				param.shrinking = atoi(argv[i]);
				break;
			case 'b':
				param.probability = atoi(argv[i]);
				break;
			case 'q':
				print_func = &print_null;
				i--;
				break;
			case 'v':
				cross_validation = 1;
				nr_fold = atoi(argv[i]);
				if(nr_fold < 2)
				{
					fprintf(stderr,"n-fold cross validation: n must >= 2\n");
					exit_with_help();
				}
				break;
			case 'w':
				++param.nr_weight;
				param.weight_label = (int *)realloc(param.weight_label,sizeof(int)*param.nr_weight);
				param.weight = (double *)realloc(param.weight,sizeof(double)*param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
				param.weight[param.nr_weight-1] = atof(argv[i]);
				break;
			default:
				fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
		}
	}
	
	svm_set_print_string_function(print_func);
	
	// determine filenames
	
	if(i>=argc)
		exit_with_help();
	
	strcpy(input_file_name, argv[i]);
	
	if(i<argc-1)
		strcpy(model_file_name,argv[i+1]);
	else
	{
		char *p = strrchr(argv[i],'/');
		if(p==NULL)
			p = argv[i];
		else
			++p;
		sprintf(model_file_name,"%s.model",p);
	}
}



// read in a problem (in svmlight format)

void read_problem(const char *filename)
{
	int elements, max_index, inst_max_index, i, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;
	
	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}
	
	prob.l = 0;
	elements = 0;
	
	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label
		
		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			++elements;
		}
		++elements;
		++prob.l;
	}
	rewind(fp);
	

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct svm_node *,prob.l);
	x_space = Malloc(struct svm_node,elements);
	
	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);
		
		prob.y[i] = strtod(label,&endptr);

		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);
		
		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");
			
			if(val == NULL)
				break;
			
			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = x_space[j].index;
			
			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);
			
			++j;
		}
		
		if(inst_max_index > max_index)
			max_index = inst_max_index;
		x_space[j++].index = -1;
	}

	prob.max_index = max_index;
	if(param.gamma == 0 && max_index > 0)
		param.gamma = 1.0/max_index;
		
	fclose(fp);
}


