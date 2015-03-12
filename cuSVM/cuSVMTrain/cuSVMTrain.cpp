
#include <stdio.h>
#include <math.h>


#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

extern "C"
void SVMTrain(float *mexalpha,float* beta,float*y,float *x ,float C, float kernelwidth, int m, int n, float StoppingCrit);

extern "C"
void SVRTrain(float *mexalpha,float* beta,float*y,float *x ,float C, float kernelwidth, float eps, int m, int n, float StoppingCrit);




#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include "svm.h"
#include <math.h>
#include <string>
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

void print_null(const char *s) {}

void exit_with_help()
{
	printf(
		"Usage: svm-train [options] training_set_file [model_file]\n"
		"options:\n"
		"-s svm_type : set type of SVM (default 0)\n"
		"	0 -- C-SVC		(multi-class classification)\n"
		"	1 -- nu-SVC		(multi-class classification)\n"
		"	2 -- one-class SVM\n"
		"	3 -- epsilon-SVR	(regression)\n"
		"	4 -- nu-SVR		(regression)\n"
		"-t kernel_type : set type of kernel function (default 2)\n"
		"	0 -- linear: u'*v\n"
		"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
		"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
		"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
		"	4 -- precomputed kernel (kernel values in training_set_file)\n"
		"-d degree : set degree in kernel function (default 3)\n"
		"-g gamma : set gamma in kernel function (default 1/num_features)\n"
		"-r coef0 : set coef0 in kernel function (default 0)\n"
		"-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
		"-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
		"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
		"-m cachesize : set cache memory size in MB (default 100)\n"
		"-e epsilon : set tolerance of termination criterion (default 0.001)\n"
		"-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
		"-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
		"-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
		"-v n: n-fold cross validation mode\n"
		"-q : quiet mode (no outputs)\n"
		"-l walltime : set maximum walltime in minutes (default -1)\n"
		"-a savetime : set time interval in minutes to report current primalvalue (default -1)\n"
		"-x modelpath : path to save walltime models (default .).\n"
		"-k subsampling: amount to subsample, between 0.0 and 1.0, e.g. 0.5=half of dataset. First N elements will be taken. (default 1.0)\n"
	);
	exit(1);
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
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


extern double subsamplingAmount;
extern int walltime;
extern int savetime;
extern std::string modelPath;


int main(int argc, char **argv)
{
	char input_file_name[1024];
	char model_file_name[1024];
	const char *error_msg;
	
	parse_command_line(argc, argv, input_file_name, model_file_name);
	read_problem(input_file_name);
	error_msg = svm_check_parameter(&prob,&param);
	
	if(error_msg)
	{
		fprintf(stderr,"ERROR: %s\n",error_msg);
		exit(1);
	}
	
	if(cross_validation)
	{
		do_cross_validation();
	}
	else
	{
		model = svm_train(&prob,&param);
		if(svm_save_model(model_file_name,model))
		{
			fprintf(stderr, "can't save model to file %s\n", model_file_name);
			exit(1);
		}
		svm_free_and_destroy_model(&model);
	}
	svm_destroy_param(&param);
	free(prob.y);
	free(prob.x);
	free(x_space);
	free(line);
	
	return 0;
}

void do_cross_validation()
{
	int i;
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double *target = Malloc(double,prob.l);
	
	svm_cross_validation(&prob,&param,nr_fold,target);
	if(param.svm_type == EPSILON_SVR ||
		param.svm_type == NU_SVR)
	{
		for(i=0;i<prob.l;i++)
		{
			double y = prob.y[i];
			double v = target[i];
			total_error += (v-y)*(v-y);
			sumv += v;
			sumy += y;
			sumvv += v*v;
			sumyy += y*y;
			sumvy += v*y;
		}
		printf("Cross Validation Mean squared error = %g\n",total_error/prob.l);
		printf("Cross Validation Squared correlation coefficient = %g\n",
			   ((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
			   ((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
		);
	}
	else
	{
		for(i=0;i<prob.l;i++)
			if(target[i] == prob.y[i])
				++total_correct;
			printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
	}
	free(target);
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
	walltime = -1;
	savetime = -1;
	cross_validation = 0;
	subsamplingAmount  = -1;
	
	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 'k':
				subsamplingAmount = atof(argv[i]);
				if ((subsamplingAmount <= 0.0) || (subsamplingAmount > 1.0))
				{
					fprintf(stderr,"Subsampling amount must be > 0.0 and <= 1.0!\n");
					exit_with_help();
				}
				break;
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
			case 'a':
				savetime = atoi(argv[i]);
				break;
			case 'l':
				walltime = atoi(argv[i]);
				break;
			case 'x':
				modelPath = (argv[i]);
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
	
	if (subsamplingAmount != -1)
	{
		// compute new size of subsampled dataset
		unsigned int subsampledSize = (unsigned int)floor((double) prob.l*subsamplingAmount);
		
		// sanity check
		if (subsampledSize > (unsigned int) prob.l) 
		{
			printf("WARNING: Subsampling yielded more elements than read. Disabling Subsampling!\n");
			subsampledSize = prob.l;
		}
		if (subsampledSize == 0)
		{
			fprintf(stderr,"Subsampling yielded zero elements! No data, no SVM!\n");
			exit(1);
		}
		
		printf("Found %u data points, reading the first %u of them.", (unsigned int)  prob.l, subsampledSize);
		prob.l = (unsigned int) subsampledSize;
	}
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
	
	if(param.gamma == 0 && max_index > 0)
		param.gamma = 1.0/max_index;
	
	if(param.kernel_type == PRECOMPUTED)
		for(i=0;i<prob.l;i++)
		{
			if (prob.x[i][0].index != 0)
			{
				fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
				exit(1);
			}
			if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			{
				fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
				exit(1);
			}
		}
		
		fclose(fp);
}




/*


//nlhs -Number of expected output mxArrays
//
//plhs - Array of pointers to the expected output mxArrays
//
//nrhs - Number of input mxArrays
//
//prhs - Array of pointers to the input mxArrays. Do not modify any prhs values in your MEX-file. 
//		Changing the data in these read-only mxArrays can produce undesired side effects.
//[alphas,beta,svs]=cuSVMTrain(y,train,C,kernel,eps, (optional) stoppingcrit)
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )

{

	
	if (nlhs>3)
		mexErrMsgTxt("cuSVMTrain has at most 3 outputs.");
	
	if (nrhs>6)
		mexErrMsgTxt("Too many input arguments.");

	if (nrhs<5)
		mexErrMsgTxt("Too few input arguments.");

	//prhs[0] === y   prhs[1]===train
	if (mxIsClass(prhs[0], "single") + mxIsClass(prhs[1], "single")!=2)
		mexErrMsgTxt("Both the target vector and feature matrix must consist of single precision floats.");
	
	//prhs[1]===train , okreœlamy wymiary macierzy z elementami treningowymi, lecz czy ona jest rzadka?
	int n=mxGetN(prhs[1]);
	int m=mxGetM(prhs[1]);
	

	if (mxGetM(prhs[0])!=m)
		mexErrMsgTxt("The target vector and feature matrix must have the same number of rows.");

	if (mxGetN(prhs[0])!=1)
		mexErrMsgTxt("The target vector must only have one column.");

	//prhs[2] === C prhs[3]===kernel (gamma)
	if ((mxGetM(prhs[2])!=1) | (mxGetN(prhs[2])!=1)|(mxGetM(prhs[3])!=1) | (mxGetN(prhs[3])!=1)|((nrhs>=5&&(mxIsEmpty(prhs[4])!=1))?(mxGetM(prhs[4])!=1) | (mxGetN(prhs[4])!=1):0)|(nrhs==6?(mxGetM(prhs[5])!=1) | (mxGetN(prhs[5])!=1):0))
		mexErrMsgTxt("The regularization parameter (C), the kernel width, epsilon, and the stopping criterion (if specified) all must be scalars.");

	



	float C;
	float kernelwidth;
	float eps;
	 

	int IsRegression=0;

	if (mxIsEmpty(prhs[4])!=1)
	{	
		IsRegression=1;
		
		if (mxIsClass(prhs[4],"double")==1)
			eps=(float)*(double *)mxGetData(prhs[4]);
		else if (mxIsClass(prhs[4],"single")==1)
			eps=*(float *)mxGetData(prhs[4]);
		else
			mexErrMsgTxt("The regularization parameter (C), the kernel width, epsilon, and the stopping criterion (if specified) all must be either single or double precision floats.");
	}
	
	float StoppingCrit=0.001;
	
	if (nrhs==6)
	{

		if (mxIsClass(prhs[5],"double")==1)
			StoppingCrit=(float)*(double *)mxGetData(prhs[5]);
		else if (mxIsClass(prhs[5],"single")==1)
			StoppingCrit=*(float *)mxGetData(prhs[5]);
		else
			mexErrMsgTxt("The regularization parameter (C), the kernel width, epsilon, and the stopping criterion (if specified) all must be either single or double precision floats.");

		if ((StoppingCrit<=0) | (StoppingCrit>=.5) )
			mexErrMsgTxt("The stopping criterion must be greater than zero and less than .5.");

	}	


	float* y=(float *)mxGetData(prhs[0]);
	float* x=(float *)mxGetData(prhs[1]);

	plhs[1]=mxCreateNumericMatrix(1, 1,mxSINGLE_CLASS, mxREAL);
	float *beta=(float*)mxGetData(plhs[1]);

	float *alpha=new float [m];

	if (IsRegression)
	{

		SVRTrain(alpha,beta,y,x ,C,kernelwidth,eps ,m,n,StoppingCrit);
	}
	else
	{
		int JustOneClassError=1;
		int NotOneorNegOneError=0;
		float FirstY=y[0];

		for(int k=0;k<m;k++)
		{
			if(y[k]!=FirstY) {JustOneClassError=0;}
			if((y[k]!=1.0) && (y[k]!=-1.0) ){NotOneorNegOneError=1;}
		}
		
		if (JustOneClassError==1)
			mexErrMsgTxt("All training labels are of the same class.  There must of course be two classes");
		
		if (NotOneorNegOneError==1)
			mexErrMsgTxt("Training labels must be either 1 or -1.");

		
		SVMTrain(alpha,beta,y,x ,C,kernelwidth,m,n,StoppingCrit);
		
	}
	
	
	int numSVs=0;
	int numPosSVs=0;
	for(int k=0;k<m;k++)
	{
		if(alpha[k]!=0)
		{
			if(IsRegression==0) 
			{
				alpha[k]*=y[k];
				if(y[k]>0) {numPosSVs++;}
			}
			
			numSVs++;
		}
	}
	
	
	plhs[0]=mxCreateNumericMatrix(numSVs, 1,mxSINGLE_CLASS, mxREAL);
	float *SvAlphas=(float*)mxGetData(plhs[0]);
	
	plhs[2]=mxCreateNumericMatrix(numSVs, n,mxSINGLE_CLASS, mxREAL);
	float *Svs=(float*)mxGetData(plhs[2]);

	

	if(IsRegression==0) 
	{
	
		int PosSvIndex=0;
		int NegSvIndex=0;

		for(int k=0;k<m;k++)
		{
			if(alpha[k]!=0)
			{
				if(y[k]>0)
				{
					SvAlphas[PosSvIndex]=alpha[k];
					for(int j=0;j<n;j++)
					{Svs[PosSvIndex+j*numSVs]=x[k+j*m];}
					PosSvIndex++;
				}
				else
				{
					SvAlphas[NegSvIndex+numPosSVs]=alpha[k];
					for(int j=0;j<n;j++)
					{Svs[NegSvIndex+numPosSVs+j*numSVs]=x[k+j*m];}
					NegSvIndex++;
				}			
			}		
		}
	}
	else
	{
		int svindex=0;

		for(int k=0;k<m;k++)
		{
			if(alpha[k]!=0)
			{
				SvAlphas[svindex]=alpha[k];
				for(int j=0;j<n;j++)
				{Svs[svindex+j*numSVs]=x[k+j*m];}
				svindex++;
			}
		
		}

	 }



	return;
}
*/