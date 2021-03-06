% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/salso.R
\name{computeAdjacencyMatrix}
\alias{computeAdjacencyMatrix}
\title{Convert partition labels to adjacency matrix}
\usage{
computeAdjacencyMatrix(labels)
}
\arguments{
\item{labels}{The integer vector of partition labels.}
}
\value{
A binary integer matrix \eqn{E} where 
\eqn{E_{ij} = 1} if \eqn{i} and \eqn{j} if
item i and item j have the same partition label, and 0 otherwise.
}
\description{
Convert partition labels to adjacency matrix
}
