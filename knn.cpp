#include <Rcpp.h>
#include <math.h>
#include <iostream>
#include <vector>

using namespace Rcpp;

// [[Rcpp::plugins(cpp11)]]

int runKnn(const NumericVector &row, const NumericMatrix &trainSet, const IntegerVector &labels, const unsigned int k, const double p);
double minkowskiDistance(const double & p, const NumericVector & ds);

IntegerVector prepareAnswerVector(IntegerVector &labels, unsigned int length);
unsigned int checkDParameter(NumericMatrix &a, NumericMatrix &b);
int findRandomMax(std::vector<int> &occurences);

// [[Rcpp::export]]
/**
 * Runs knn algorithm
 * @param trainSet Training set, real matrix n x d, representing n points in R^d
 * @param labels Labels for training set, factor type, size must be n
 * @param testSet Real matrix m x d 
 * @param k Nearest neighbours, 1 <= k <= n
 * @param p Parameter for Minkowski's metric, p >= 1, 2 by default
 * @return Labels for testSet
 */
IntegerVector knn(NumericMatrix trainSet, IntegerVector labels, NumericMatrix testSet, const unsigned int k, const double p = 2) {
  unsigned int n = trainSet.nrow();
  unsigned int m = testSet.nrow();
  checkDParameter(trainSet, testSet);
  
  if(n != labels.size())
  {
    stop("Number of labels is not equal to number of training points");
  }
  
  if(k > n || k < 1)
  {
    stop("Wrong k value");
  }
  
  if(p < 1)
  {
    stop("p must be >= 1");
  }
  
  IntegerVector result = prepareAnswerVector(labels, m);
  
  //for each point from training set
  for(unsigned int i = 0; i < m; i++)
  {
    //run knn
    result[i] = runKnn(testSet.row(i), trainSet, labels, k, p);
  }
  
  return result;
}

/**
 * Find label for one vector (one row)
 * @param row Input vector
 * @param trainSet Training data
 * @param labels Matching labels for training data
 * @param k Number of nearest neighbours
 * @param p Parameter for Minkowski's metric
 * @return Label for input vector
 */
int runKnn(const NumericVector &row, const NumericMatrix &trainSet, const IntegerVector &labels, const unsigned int k, const double p)
{
  unsigned int n = trainSet.nrow();
  unsigned int l = labels.size();
  
  std::vector< std::pair<double, int> > distances(n);
  
  //find distance to each point from train set
  for(unsigned int i = 0; i < n; i++)
  {
    double distanceToI = minkowskiDistance(p, row - trainSet.row(i));
    //first - distance, second - label from training set
    distances[i] = std::make_pair(distanceToI, labels[i]);
  }
  
  //sort distances so that nearest are first in vector
  std::sort(distances.begin(), distances.end());
  
  //take first k elements from distances and find most frequent laber (second in pair)
  //assume labels are from 1 to l
  std::vector<int> occurences(l, 0);
  for(unsigned int i = 0; i < k; i++)
  {
    //map 1. label to 0 element in occurences etc, increment (determining frequency)
    occurences[distances[i].second - 1]++;
  }

  return findRandomMax(occurences);
}


IntegerVector prepareAnswerVector(IntegerVector &labels, const unsigned int length)
{
  IntegerVector w(length);
  w.attr("levels") = labels.attr("levels");
  w.attr("class") = "factor";
  return w;
}

/**
 * Check if d matches (size of one point)
 */
unsigned int checkDParameter(NumericMatrix &trainSet, NumericMatrix &testSet)
{
  unsigned int dTrain = trainSet.ncol();
  unsigned int dTest = testSet.ncol();
  
  if(dTrain != dTest) 
  {
    stop("d not consistent in test set and train set");
  }
  
  return dTrain;
}
/**
 * Given occurences count, find most frequent value (index starting from 1). If many, pick randomly.
 */
int findRandomMax(std::vector<int> &occurences)
{
  int max = *std::max_element(occurences.begin(), occurences.end());
  std::vector<int> idx;
  
  for(unsigned int i = 0; i < occurences.size(); i++)
  {
    if(occurences[i] == max)
    {
      idx.push_back(i + 1);
    }
  }
  
  //select randomly from max
  NumericVector xx = runif(1, 0, idx.size());
  return idx[(int) xx[0]];
}

/**
 * Author: Derrick Pallas
 * License: zlib
 * https://gist.githubusercontent.com/pallas/5565528/raw/fb290857204185b8a74ad585d7250f84b4df8ff2/minkowski_distance.cc
 */
double minkowskiDistance(const double & p, const NumericVector & ds) {
  double ex = 0.0;
  double min_d = std::numeric_limits<double>::infinity();
  double max_d = -std::numeric_limits<double>::infinity();
  for (unsigned int i = 0 ; i < ds.size() ; ++i) {
    double d = std::fabs(ds[i]);
    ex += std::pow(d, p);
    min_d = std::min(min_d, d);
    max_d = std::max(max_d, d);
  }
  
  return std::isnan(ex) ? ex
    : !std::isnormal(ex) && std::signbit(p) ? min_d
    : !std::isnormal(ex) && !std::signbit(p) ? max_d
    : std::pow(ex, 1.0/p);
}
