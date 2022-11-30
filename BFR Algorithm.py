import numpy as np
import math
from math import floor
from random import sample
from sklearn.cluster import KMeans
from itertools import combinations
from math import sqrt
import sys

class BFR(object):
    def __init__(self,data,clusters):
        self.X = data[:,2:]
        self.dimension = self.X.shape[1]
        self.index = list(i for i in range(data.shape[0]))
        self.clusters = clusters
        self.data = data
    
    def random_sampling(self,index,sampleLength):
        """
        :parameter size of sample
        output: sampleIndexes, sampleData
        """
        sampleIndex = sample(index,sampleLength)
        return sampleIndex,self.data[sampleIndex,:]

    def Kmeans(self,algoType,sampleIndex):
        """
        :parameter sampleIndex based on the population
        output: A dictionary with cluster label and corresponding sample_indexes as list
        """
        n_clusters = self.clusters
        if algoType == "large":
            #run K-means with large K (e.g., 5 times of the number of the parameter clusters)
            clusters = {}
            if len(sampleIndex) <= self.clusters * 5:
                clusters = {i:[index] for i,index in enumerate(sampleIndex)}
                return clusters
            else:
                n_clusters = self.clusters * 5

        kmeans = KMeans(n_clusters= n_clusters,random_state=0).fit(self.X[sampleIndex, :])  
        y = kmeans.labels_
        clusters = {k :[] for k  in np.unique(y)}
        for i in range(len(y)):
            clusters[y[i]].append(sampleIndex[i])

        return clusters

    def singlePoints(self,clusterDictionary):
        """
        :parameter cluster Dictionary
        output: indexes of clusters with single points and the clusters with more than single points
        """
        singlePoints,singleIndex = [],[]
        for k in clusterDictionary.keys():
            if len(clusterDictionary[k]) == 1:
                singlePoints.append(k)
                singleIndex += clusterDictionary[k]
        clusterDictionary = {k:clusterDictionary[k] for k in clusterDictionary.keys() if k not in singlePoints}
        return singleIndex,clusterDictionary
    
    def statsInformation(self,clusterDictionary):
        """
        :parameter clusterDictionary
        output: information required for Discard Set/Compression Set Cluster
        """
        dsDictionary = {}
        for k,val in clusterDictionary.items():
            numberOfPoints = len(val)
            Data = self.X[val,:]
            sumationVector = Data.sum(axis=0)
            sumSquaredVector = (Data*Data).sum(axis=0)
            dsDictionary[k] = [numberOfPoints,sumationVector,sumSquaredVector]
        return dsDictionary

    def Mahalanobis_distance(self,point,clusterInfo):
        """
        :parameter point
        :parameter cluster stat information
        output: Mahalanobis_distance
        """
        avg = clusterInfo[1]/clusterInfo[0]
        stdVector = np.sqrt((clusterInfo[2]/clusterInfo[0]) - avg**2)
        return sqrt(sum(((point - avg)/stdVector) ** 2))

    def checkPoint(self,index,dsDictionary):
        """
        :parameter index
        :parameter dsDictionary:
        output: cluster index
        """
        pointvector = self.X[index,:]
        leastDistance = 10000 
        clusterIndex = -1
        if dsDictionary:
            for key,vector in dsDictionary.items():
                self.Mahalanobis_distance(pointvector, vector)
                dist = self.Mahalanobis_distance(pointvector,vector)
                if dist < leastDistance and dist < math.sqrt(self.dimension)*2:
                    leastDistance,clusterIndex = dist,key
        return clusterIndex

    def assignPoints(self,newPoints,dsDictionary,csDictionary):
        """
        :parameter newPoints: index of points that needs to be assigned
        :parameter dsDictionary
        :parameter csDictionary: current cs summary
        output: two dict, 1 list
        """
        #get cluster index and list of points as dictionary
        dsPoints = {k:[] for k in dsDictionary.keys()}
        csList =[]
        #get cluster index and list of points as dictionary
        csPoints = ({k:[] for k in csDictionary.keys()} if csDictionary else {})
        rsPoints = []
    
        for point in newPoints:
            # check if the point could be assign to some of those sets
            cluster = self.checkPoint(point,dsDictionary)
            if cluster != -1:
                dsPoints[cluster].append(point)
            else:
                csList.append(point)
        
        if csList and csDictionary:
            for  point  in  csList:
                cluster  =  self.checkPoint( point, csDictionary)
                if cluster  !=  -1:
                    csPoints[cluster].append( point)
                else:
                    rsPoints.append( point)
        elif csList:
            rsPoints += csList

        return dsPoints,csPoints,rsPoints

    def renewDictionary(self,pointDictionary,dsDictionary,dsCluster):
        if pointDictionary:
            for key,points in pointDictionary.items():
                summary = [len(points),sum(self.X[index,:] for index in points),sum(self.X[index,:]*self.X[index,:] for index in points)]
                dsCluster[key] += points
                dsDictionary[key] = [dsDictionary[key][k]+summary[k] for k in range(3)]
        return dsDictionary, dsCluster
    
    def mahalanobisDistanceForCluster(self,cluster1,cluster2):
        """
        :parameter cluster1
        :parameter cluster2
        output: mahalanobis distance
        """
        return self.Mahalanobis_distance(cluster1[1]/cluster1[0],cluster2)

    def mergeCluster(self,cluster1,cluster1Points,cluster2,cluster2Points):
        """
        :parameter cluster1
        :parameter cluster1Points: the index of points
        :parameter cluster2:
        :parameter cluster2Points: the index of points 
        output: one Cluster where no two clusters has mahalanobis distance < 2*sqrt(d)
        """
        if len(cluster1) != 0 and len(cluster2) != 0:
            csCluster2Adjacency = { key+max(cluster1.keys()):cluster2[key] for key in cluster2}
            mergedCluster = { **cluster1, **csCluster2Adjacency} 
            cluster2PointsAdjacency = { key+max(cluster1Points.keys()):cluster2Points[key] for key in cluster2}
            mergedClusterPoints = { **cluster1Points, **cluster2PointsAdjacency}
            d = self.dimension
            while True:
                min_dist,drop_set  = math.inf,tuple()
                for (set1,set2) in combinations(mergedCluster.keys(),2):
                    dist = self.mahalanobisDistanceForCluster(mergedCluster[set1],mergedCluster[set2])
                    if dist <=min_dist:
                        drop_set = (set1,set2)
                        min_dist = dist
                if min_dist > math.sqrt(d)*2:
                    break
                else:
                    set1,set2 = drop_set
                    mergedCluster[set1][0] += mergedCluster[set2][0]
                    mergedCluster[set1][2] += mergedCluster[set2][2]
                    mergedCluster[set1][1] += mergedCluster[set2][1]
                    mergedCluster.pop(set2)
                    mergedClusterPoints[set1] += mergedClusterPoints[set2]
                    mergedClusterPoints.pop(set2)
            return mergedCluster,mergedClusterPoints
        elif len(cluster2) != 0:
            return cluster2,cluster2Points
        elif len(cluster1) != 0:
            return cluster1,cluster1Points
        else:
            return {},{}

    def fit(self):
        remainingPoints = self.index  
        length = self.data.shape[0]
        sampleLength = floor(length/5)
        summary = [] 
        remainingPoints = self.index
        # Load 20% of the data randomly.
        staringInd, _ = self.random_sampling(self.index,sampleLength)
        #step2
        # Run K-Means with a large K on the data in memory using the Euclidean distance as the similarity measurement.
        step2 = self.Kmeans("large",staringInd)
        #step3
        # In the K-Means result from Step 2, move all the clusters that contain only one point to RS (outliers).
        step3Points, _ = self.singlePoints(step2)
        #step4
        # initial discard point set
        dsPoints = [index for index in staringInd if index not in step3Points]
        # Run K-Means again to cluster the rest of the data points with K = the number of input clusters.
        intialCluster = self.Kmeans("normal",dsPoints)
        # step 5  Use the K-Means result from Step 4 to generate the DS clusters (i.e., discard their points and generate statistics).
        dsDictionary = self.statsInformation(intialCluster)
        dsCluster = intialCluster
        # The initialization of DS has finished, so far, you have K numbers of DS clusters (from Step 5) and some numbers of RS (from Step 3).

        # step 6
        # Run K-Means on the points in the RS with a large K to generate CS (clusters with more than one points) and RS (clusters with only one point).
        intermediateCluster = self.Kmeans("large",step3Points)
        rs, csCluster = self.singlePoints(intermediateCluster)
        csDictionary = self.statsInformation(csCluster)
        

        # summuraize: number of discard points, number of clusters in compression set, number of the compression points, number of points in Retained set”
        summary.append(['Round 1:', len(dsPoints),(len(csDictionary.keys) if csDictionary else 0),len(step3Points) - len(rs),len(rs)])
        remainingPoints = list(set(remainingPoints) - set(staringInd))

        iteration = 4
        while len(remainingPoints) > 0 :
            sampleLength = len(remainingPoints) // iteration
            #step7 
            # Load another 20% of the data randomly.
            sampleIndex,_ = self.random_sampling(remainingPoints,sampleLength)
            remainingPoints = list(set(remainingPoints) - set(sampleIndex))
            iteration = iteration - 1
            
            dsPoints, csPoints, rsPoints = self.assignPoints(sampleIndex,dsDictionary,csDictionary)

            #Step 8
            # For new points, compare them to each of DS using Mahalanobis Distance and assign them to nearest DS clusters if distance is < 2√d.
            dsDictionary,dsCluster = self.renewDictionary(dsPoints,dsDictionary,dsCluster)

            #Step 9. 
            # For new points that are not assigned to DS clusters, using Mahalanobis Distance and assign these points to nearest CS clusters if distance is < 2√d
            csDictionary,csCluster = self.renewDictionary(csPoints,csDictionary,csCluster)

            #Step 10
            # For the new points that are not assigned to a DS cluster or a CS cluster, assign them to RS.
            rs += rsPoints

            #Step 11
            # Run K-Means on the RS with a large K to generate CS (clusters with more than one points) and RS (clusters with only one point).
            newCs = self.Kmeans("large",rs)
            rs,newCsCluster = self.singlePoints(newCs)
            newCsDictionary = self.statsInformation(newCsCluster)#get the renew rs, and new cs dict
            # step 12
            # Merge CS clusters that have a Mahalanobis Distance < 2√d.
            csDictionary,csCluster = self.mergeCluster(csDictionary,csCluster,newCsDictionary,newCsCluster)
            summary.append([f'Round {5-iteration}:',sum(val[0] for val in dsDictionary.values()),(len(csDictionary.keys()) if csDictionary else 0),sum(val[0] for val in csDictionary.values()),len(rs)])
        # last run
        #If this is the last run , merge CS clusters with DS clusters that have a Mahalanobis Distance < 2√d.
        _,dsPoints = self.mergeCluster(dsDictionary,dsCluster,csDictionary,csCluster)
        out = list()
        for id,indexes in dsPoints.items():
            if id > self.clusters: # those are retained number
                for num in indexes:
                    out.append((num,-1))
            else:
                for num in indexes:
                    out.append((num,id))

        return summary,sorted(out,key=lambda x: x[0])

if __name__ == "__main__":

    infile = sys.argv[1]
    clusters = int(sys.argv[2])
    outfile = sys.argv[3]

    Data = np.genfromtxt(infile,delimiter=',')
    Data[:,0] = Data[:,0].astype(np.int32)
    BFRAlgo = BFR(Data,clusters)
    summary, output = BFRAlgo.fit()


    with open(outfile,'w+') as file:
        file.write("The intermediate results:"+"\n")
        for line in summary:
            file.write(line[0] + ','.join(str(line[i]) for i in range(1,5)) + '\n')
        file.write("\nThe clustering results:\n")
        for item in output:
            file.write(','.join([str(i) for i in item]) + "\n")
    file.close()
