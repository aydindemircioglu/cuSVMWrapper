# copied and modified from wuSVM
# 	by aydin demircioglu, aydin.demircioglu <at ini.rub.de>


# http://link.springer.com/chapter/10.1007%2F978-3-540-87477-5_45


	cmake_minimum_required (VERSION 2.8)
	project (cuSVM)

	#Set the version number
	set (cuSVM_VERSION_MAJOR 0)
	set (cuSVM_VERSION_MINOR 1)


	set (cuSVMTrainDir "${PROJECT_SOURCE_DIR}/cuSVMTrain")
	set (cuSVMPredictDir "${PROJECT_SOURCE_DIR}/cuSVMPredict")
	set (cuSVMIncludeDir "${PROJECT_SOURCE_DIR}/inc")

	include_directories("${PROJECT_BINARY_DIR}")
	include_directories(cuSVMTrainDir)
	include_directories(cuSVMPredictDir)
	include_directories(cuSVMIncludeDir)

	#


#find_package(BLAS REQUIRED)
#find_package(LAPACK REQUIRED)



# better do not turn that off.
	option (USE_CUDA "Build with CUDA acceleration?" ON)

# really really check cuda	
	if (USE_CUDA)
		find_package(CUDA 5.0)
		if(NOT CUDA_FOUND)
			message(FATAL_ERROR "CUDA not found!")
		endif()
	endif()

# and make stress if not found
	if(NOT CUDA_FOUND)
		message(FATAL_ERROR "Did not find CUDA.. Sorry, no CUDA, no play.")
	endif ()
	
	
	
# other options
	option (DEBUG_MODE "Build in debug mode?" OFF)
	if(DEBUG_MODE)
		set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
		set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -G")
	else()
		set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
		add_definitions(-DNDEBUG)
	endif()

	option(USE_GPROF "Build with gprof profiling information" OFF)
	if(USE_GPROF)
		set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pg")
		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
	endif()

	
	
# nvcc compile things	
	add_definitions(-DCUDA)
	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_30,code=sm_30)
	cuda_compile(cuSVMSolver_O 
		${cuSVMIncludeDir}/cuSVMutil.h
		${cuSVMTrainDir}/cuSVMSolver.cu
		)
	cuda_compile(cuSVMPredictKernel_O 
		${cuSVMIncludeDir}/cuSVMutil.h
		${cuSVMPredictDir}/cuSVMPredictKernel.cu
	)
	set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)


# executable	
	cuda_add_executable(cuSVMTrain 
		${cuSVMTrainDir}/cuSVMTrain.cpp 
	#	${cuSVMPredictDir}/svm.cpp
		${cuSVMSolver_O} 
	)
	CUDA_ADD_CUBLAS_TO_TARGET( cuSVMTrain )

	cuda_add_executable(cuSVMPredict 
		${cuSVMPredictDir}/cuSVMPredict.cpp 
		${cuSVMPredictKernel_O} 
		${cuSVMPredictDir}/svm.cpp
	)
	CUDA_ADD_CUBLAS_TO_TARGET( cuSVMPredict )


# link things	
	target_link_libraries(cuSVMTrain ${BLAS_LINKER_FLAGS} ${BLAS_LIBRARIES})
	target_link_libraries(cuSVMTrain ${LAPACK_LINKER_FLAGS} ${LAPACK_LIBRARIES})
	target_link_libraries(cuSVMTrain cuda)

	target_link_libraries(cuSVMPredict ${BLAS_LINKER_FLAGS} ${BLAS_LIBRARIES})
	target_link_libraries(cuSVMPredict ${LAPACK_LINKER_FLAGS} ${LAPACK_LIBRARIES})
	target_link_libraries(cuSVMPredict cuda)

	
# install things	
	install (TARGETS cuSVMTrain DESTINATION bin)
	install (TARGETS cuSVMPredict DESTINATION bin)

	include (InstallRequiredSystemLibraries)
	#set (CPACK_RESOURCE_FILE_LICENSE  
	#    "${CMAKE_CURRENT_SOURCE_DIR}/../license.txt")
	set (CPACK_PACKAGE_VERSION_MAJOR "${SP_SVM_VERSION_MAJOR}")
	set (CPACK_PACKAGE_VERSION_MINOR "${SP_SVM_VERSION_MINOR}")

	include (CPack)

	
	
	