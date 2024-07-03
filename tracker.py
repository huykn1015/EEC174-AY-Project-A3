from math import sqrt


class TrackedObject:
    count = 0
    def __init__(self, centroid):
        self.object_id = TrackedObject.count
        TrackedObject.count += 1
        self.path = [centroid]
        self.lost_count = 0
    def update(self, centroid):
        self.last_pos = centroid
        self.path.append(centroid)
        self.lost_count = 0
    def inc_lost(self):
        self.lost_count += 1
    def dec_lost(self):
        self.lost_count -= 1
    def lost_frames(self):
        return self.lost_count
    def get_id(self):
        return self.object_id
    def get_path(self):
        return self.path
    def last_pos(self):
        return self.path[-1]



class Tracker:
    MAX_DISTANCE = 50
    def __init__(self, maxLost=30):
        """
        Initialize the tracker.

        Parameters:
        maxLost (int): Maximum number of frames an object can be lost before deregistration.
        """
        self.objects = 0
        self.maxLost = maxLost
        self.object_ids = [] # [object_id]
        #self.objects = {} # {object_id : TrackedObject}
        self.last_pos = {} # {centroid : TrackedObject}
        
        pass

    def register(self, centroid):
        """
        Register a new object with the next available ID.

        Parameters:
        centroid: The centroid of the new object to register (x, y).

        Returns:
        None
        """
        new_object = TrackedObject(centroid)
        self.object_ids.append(new_object.get_id())
        self.last_pos[centroid] = new_object
        return new_object

    def deregister(self, objectID):
        """
        Deregister an object, removing it from tracking.

        Parameters:
        objectID: The ID of the object to deregister.

        Returns:
        None
        """
        print(objectID)
        print(self.object_ids)
        self.object_ids.remove(objectID)
        rm_centroid = None
        for c, obj in self.last_pos.items():
            if obj.get_id() == objectID:
                rm_centroid = c

        self.last_pos.pop(rm_centroid)


        pass

    def update(self, inputCentroids):
        #print('---------------')
        """
        Update the tracked objects with new centroid information from the current frame.

        Parameters:
        inputCentroids: list of centroids detected in the current frame.

        Returns:
        Updated objects with their current centroid positions.
        """
        inputCentroids = list(inputCentroids)
        res = []
        if len(self.object_ids) == 0: #if no tracked objects, register all centroids
            for centroid in inputCentroids:
                new_object = self.register(centroid)
                res.append((new_object.get_id(), (int(centroid[0]), int(centroid[1]))))
            return res
        tracked_centroids = list(self.last_pos.keys())
        dist = self._distance(inputCentroids, tracked_centroids) #get distance from each new centroid to each tracked object
        identified_objects = []
        for i in range(len(inputCentroids)):
            closest_idx = dist[i].index(min(dist[i])) 
            closest_centroid = tracked_centroids[closest_idx]
            #print(tracked_centroids)
            #print(self.last_pos)
            min_dist = min(dist[i])
            if  min_dist < Tracker.MAX_DISTANCE and closest_centroid in self.last_pos and self.is_closest_centroid(dist, i,closest_idx): #register matched object
                closest_object = self.last_pos.pop(closest_centroid)
                closest_object.update(inputCentroids[i])
                identified_objects.append((inputCentroids[i], closest_object))
            else: # object does match any centroids
                self.register(inputCentroids[i])
        rm_ids = []
        for centroid in self.last_pos: #increment lost frames and remove objects lost for too long
            object = self.last_pos[centroid]
            if object.lost_frames() > self.maxLost:
                rm_ids.append(object.get_id())

            else:
                object.inc_lost()

        for id in rm_ids:
            self.deregister(id)
        for centroid, object in identified_objects: #add newly updated objects back into list of 
            self.last_pos[centroid] = object
        return [(object.get_id(), (int(centroid[0]), int(centroid[1]))) for centroid, object in identified_objects]

    def get_paths(self):
        """
        Retrieve the paths (list of centroids) of all tracked objects.

        Returns:
        Paths of tracked objects.
        """
        res = []
        for object in self.objects:
            res.append(object.get_path())
        return res
    def get_path(self, last_p):
        return self.last_pos[last_p].get_path()
    def _distance(self, a, b):
        """
        Calculate the Euclidean distances between two sets of centroids.

        Parameters:
        a: First set of centroids.
        b: Second set of centroids.

        Returns:
        Matrix of distances between each pair of centroids from set 'a' and 'b'.
        """
        #print(len(a), len(b))
        res = [[0]*len(b) for i in range(len(a))]
        #print(len(res), len(res[0]))
        for i, centroid_a in enumerate(a):
            for j, centroid_b in enumerate(b):
                xa, ya = centroid_a
                xb, yb = centroid_b
                res[i][j] = sqrt((xa-xb)**2 + (ya-yb)**2)
        return res
    def is_closest_centroid(self, dist, test_idx, c):
        col = [row[c] for row in dist]
        return test_idx == col.index(min(col))