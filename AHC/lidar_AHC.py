import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult 

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField #the types of msg 
from sensor_msgs_py import point_cloud2 #used for reading fields 
from visualization_msgs.msg import Marker, MarkerArray
# from builtin_interfaces import Duration

from sklearn.cluster import AgglomerativeClustering
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

#TODO clean this up. Some of these imports could be in "standard" use function to get them out of "lightweight"
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import timeit
from itertools import chain
from scipy.spatial.distance import cdist


class pcl2Subscriber(Node):

    def __init__(self):
        super().__init__('AHC_node')
        ## publishers and subscribers
        self.subscription = self.create_subscription(
            # message_type
            PointCloud2,
            # 'topic'
            'lidar_0/m1600/pcl2',
            self.listener_callback, 10
            )
        self.subscription #only to prevent warning for unused variable?
        self.get_logger().debug("initialized lidar_pcl2_Subscriber on lidar_0/m1600/pcl2")

        self.pub_ahc_clusters = self.create_publisher(PointCloud2, 'lidar_0/AHC/clusters', 10)
        self.pub_centroid_markers = self.create_publisher(MarkerArray, 'lidar_0/AHC/centroids', 10)

        ## ros parameters
        self.declare_parameter('AHC_framerate_goal', 5) #goal of AHC in frames per second
        self.declare_parameter('AHC_distance', 10) #[20, 40] works well for dense pcl with max distance<4m. [10-20] works for downsampled or sparse
        self.AHC_framerate_goal_ = self.get_parameter('AHC_framerate_goal').value
        self.AHC_period_ = round(1/self.AHC_framerate_goal_, 3)
        self.AHC_distance_ = self.get_parameter('AHC_distance').value 
        print('param.AHC_framerate_goal_ =', self.AHC_framerate_goal_)
        print('param.AHC_distance_  =', self.AHC_distance_)
        self.add_on_set_parameters_callback(self.parameters_callback)

        ## initialized values
        self.downsample = 10  #initial value. This is then decided by control_framerate
        self.first = 1 #flag for getting consistent labels
        

    # set and use rosparams to control framerate, etc
    def parameters_callback(self, params):
        for param in params:
            if param.name == "AHC_framerate_goal":
                if param.type_ != param.Type.INTEGER or (param.value > 30 or param.value <= 0):
                    self.get_logger().warn("AHC_framerate_goal must be integer 0 < i <= 30")
                    return SetParametersResult(successful=False)
                else:
                    self.AHC_framerate_goal_ = param.value
                    self.AHC_period_ = round(1/self.AHC_framerate_goal_, 3)

            elif param.name == "AHC_distance":
                if param.type_ != param.Type.INTEGER or (param.value > 10000 or param.value < 0):
                    self.get_logger().warn("AHC_distance must be integer 0 <= i < 10000. Recommend 10 to 50.")
                    return SetParametersResult(successful=False)
                else:
                    self.AHC_distance_ = param.value
            # print(vars(param))
            print(param.name, "=", param.value)
        return SetParametersResult(successful=True)

    # for visualization without RVIZ (mostly obsolete now)
    def plot_3d_scatter(self, x,y,z,c,cmap=cm.gist_rainbow, norm_color='scaled',figsize=(18,12),show=1,show_surface=0,cbar=1, title=''):
        ## normalize cmap to match our data. Could hardcode the thresholds so red is always a certain distance away
        if norm_color == 'labels':
            Ncolors = colors.Normalize(vmin=0,vmax=max(c))
        elif norm_color == 'fixed' or norm_color > 0:
            try:
                Ncolors = colors.Normalize(vmin=0, vmax=norm_color)
            except:
                Ncolors = colors.Normalize(vmin=0, vmax=4.1)                    
        else:  #assume norm_color == 'scaled'': 
            Ncolors = colors.Normalize(vmin=min(x),vmax=max(x))
        fig1 = plt.figure(figsize=figsize)
        ax = fig1.add_subplot(projection='3d')
        plt.title(title)
        ## scatterplot. c is the data to color along, cmap is the colorscheme
        #trying to get contours around detections
        if show_surface == 1:
            ax.scatter(x,y,z, s=30, marker=".", c=c,cmap=cmap, norm=Ncolors)
        # else it's just a normal plot
        else:
            ax.scatter(x,y,z, s=5, marker=".", c=c,cmap=cmap, norm=Ncolors)
        if cbar == 1:
            fig1.colorbar(ax.collections[0],label='distance from sensor (m)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(elev=20, azim=160)
        if show == 1:
            plt.show()      # fig1.show() or ax.show() don't work
        elif show > 0:
            plt.show(block=False) #do a timed show
            plt.pause(show)
            plt.close()
        elif show < 0:
            plt.show(block=False) # plot to screen but can't see it until they're done
            plt.pause(.0001)
        return self

    # 1. make dendogram with pcl2 msg, just to show it's working
    def plot_dendrogram(self, model, **kwargs):
        # Create linkage matrix and then plot the dendrogram
        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count
        linkage_matrix = np.column_stack( [model.children_, model.distances_, counts] ).astype(float)
        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)
        return self

    #process the point cloud here
        #filters out points if you give offsets, otherwise just formats xyz for scatterplot and AHC
    def format_pcl2(self,  min_offset=0, max_offset=0, filters=0):
        # start3 = timeit.default_timer()
        # self.get_logger().debug('start shape of xyz: %s , %s' % (np.shape(self.xyz)[0], np.shape(self.xyz)[1]))
        if filters:
            filters2 = []   
            if 'x' in filters:
                filters2.append(0)
            if 'y' in filters:
                filters2.append(1)
            if 'z' in filters:
                filters2.append(2)
            # else: 
            print('filters: ', filters, 'filters2: ', filters2)
            if min_offset != 0 or max_offset != 0:
                #pull float values out of points. From (x1,y1,z1)... to [(x1,x2...),(y1,y2...),(z1,z2...)] in xyz_array
                x = [i[0] for i in self.xyz]
                y = [i[1] for i in self.xyz]
                z = [i[2] for i in self.xyz]
                # put in array for max/min finding. Purposefully not redoing this after filtering other dimensions: 
                    #want it to be based off the initial image max/min
                xyz_array = np.array((x,y,z))
                for d in filters2:
                    min_val = min(xyz_array[d]) 
                    max_val = max(xyz_array[d])
                    #check against dimension d (x,y, or z) of x,y,z point, drop it if doesn't meet filter criteria. 
                    self.xyz = [i for i in self.xyz if ( i[d] >= (min_val + min_offset) and i[d] <= (max_val - max_offset )) ] 
                    # self.get_logger().info('after filtering on  %s,' % d)
                    self.get_logger().info('after filtering on %s, shape of xyz: %s, %s' % (d,np.shape(self.xyz)[0], np.shape(self.xyz)[1]))
            else: 
                self.get_logger().info('no filter offsets specified, skipping')
            self.get_logger().info('shape of xyz: %s , %s' % (np.shape(self.xyz)[0], np.shape(self.xyz)[1]))
        # else:
        #     self.get_logger().info('dimension to filter not specified, skipping')
        # format before returning values, whether we filtered or not
        x= [i[0] for i in self.xyz]
        y = [i[1] for i in self.xyz]
        z = [i[2] for i in self.xyz]
        # self.get_logger().info('--format time %s' %(timeit.default_timer() - start3) )
        return self, x,y,z

    # for controlling framerate/processing speed
    def decimate_pcl2(self, step_size=2):
        # breakpoint()
        # start2 = timeit.default_timer()
        self.xyz = self.xyz[::step_size]
        #breakpoint()
        #self.get_logger().info('Keeping every %sth point, shape of xyz: %s, %s' % (step_size, np.shape(self.xyz)[0], np.shape(self.xyz)[1]))
        # self.get_logger().info('decimation time %s' %(timeit.default_timer() - start2) )
        return self

    #publish labels, downsampled point clouds, etc
    def create_pcl2_and_pub(self, x,y,z, AHC_model_labels):
        fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1), 
                    PointField(name='y', offset=4, datatype=7, count=1),
                    PointField(name='z', offset=8, datatype=7, count=1),
                    PointField(name='label', offset=12, datatype=1, count=1), ]
        pt = [[x[i], y[i], z[i], AHC_model_labels[i]] for i in range(np.size(x)) ]
        header = Header()
        # header.stamp = rclpy.time.Time.now()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "lidar_0"
        pcl2_processed = point_cloud2.create_cloud(header, fields, pt)
        self.pub_ahc_clusters.publish(pcl2_processed)
        return self

    #get centroids of pcl
    def get_centroids(self, AHC_model_labels):
        label_list = np.unique(AHC_model_labels)
        centroids = []  # for centroid of envelope around object
        # c4s = []   # for weighted center using the number of points
        ## get one list with 0th item being [x0,y0,z0,label0]
        for i, t in enumerate(self.xyz):
            self.xyz[i] += (AHC_model_labels[i],)
        # get centroid of each cluster
        for label in label_list:
            filtered = [i for i in self.xyz if i[-1] == label ]
            fx = [i[0] for i in filtered]
            fy = [i[1] for i in filtered]
            fz = [i[2] for i in filtered]    
            c = ((min(fx) + max(fx)) / 2, (min(fy) + max(fy))/2, (min(fz) + max(fz))/2, label)    
            centroids.append(c)
            ## get centroids with mean of points
            # c4 = (sum(fx)/ len(fx), sum(fy)/len(fy), sum(fz)/len(fz), label)
            # c4s.append(c4)
        centroids = np.array(centroids)
        return self, centroids

    ## publish centroids of pcl as array of marker msg
    def pub_centroids(self, centroids):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "lidar_0"
        msg = MarkerArray()
        # msg.markers.append(fields)
        for i in np.arange(centroids.shape[0]):
            # ns=namespace for object name, id used for name, type of marker(0=arrow), action 0=add or modify
            fields = Marker(ns="lidar_0",id = int(i), type=0, action=0)
            fields.scale.x= .3 # maybe scale this with detection max/min bounds
            fields.scale.y = .1
            fields.scale.z = .1
            fields.color.r = min(1.0,float(centroids[i,0]/10))  #scale color with distance, hopefully
            fields.color.g = 0.2
            fields.color.b = 0.0
            fields.color.a = 1.0
            fields.lifetime = rclpy.duration.Duration(seconds=.1).to_msg()  #kill old markers after t. If a new marker with same id is published, it automatically overwrites the old one
            # fields.frame_locked = 0  #not sure about this but it might come up in TF
            fields.pose.position.x = centroids[i,0]
            fields.pose.position.y = centroids[i,1]
            fields.pose.position.z = centroids[i,2]
            # TODO orientation with PCA?
            # fields.pose.orientation.x = TODO with pca   #also y,z,w

            fields.header= header
            msg.markers.append(fields)

        # self.get_logger().info(str(fields))
        self.pub_centroid_markers.publish(msg)
        self.get_logger().debug('length of markerArray  %s' %str(len(msg.markers)))            

    def get_pub_bounding_boxes(self, AHC_model_labels):
        label_list = np.unique(AHC_model_labels)
        boxes = np.zeros(shape=(len(label_list), 24))  # bounding box will define edges with 24 points 
        # vertices = []

        ## get one list with 0th item being [x0,y0,z0,label0]
        for i, t in enumerate(self.xyz):
            self.xyz[i] += (AHC_model_labels[i],)
        # get bounds of each cluster
        for label in label_list:
            filtered = [i for i in self.xyz if i[-1] == label ]
            fx = [i[0] for i in filtered]  # turn x,y,z points into lists of x1,x2... etc
            fy = [i[1] for i in filtered]
            fz = [i[2] for i in filtered] 

            minx = min(fx)
            maxx = max(fx)
            miny = min(fy)
            maxy = max(fy)
            minz = min(fz)
            maxz = max(fz)
            # vertices
            p0 = [ minx  , miny , minz  ]
            p1 = [ maxx , miny , minz  ]
            p2 = [ maxx , maxy, minz  ]
            p3 = [ minx  , maxy, minz  ]
            p4 = [ minx  , miny , maxz ]
            p5 = [ maxx , miny , maxz ]
            p6 = [ maxx , maxy, maxz ]
            p7 = [ minx  , maxy, maxz ]
            # breakpoint()
                    #one end of box,            #other end of box,          # connecting the two
            box = [p0,p1,  p1,p2, p2,p3, p3,p0, p4,p5, p5,p6, p6,p7, p7,p4, p0,p4, p1,p5, p2,p6, p3,p7]
# TODO PICKUP HERE
            boxes[label] = box
            # boxes.append(c)
            


        
        return self, boxes


    ## this function isn't quite working as desired. Almost sure it has to do with how I 
        #backfill the centroid array with labels when the number of clusters changes. 
        # it converges towards purple and red (1/2 colors) using ex2.bag
    def match_labelnames(self, centroids_0, centroids_1, AHC_model_labels):
        self.get_logger().info("model_labels_ shape %s" %np.shape(AHC_model_labels))
        missing_rows = 0
        if np.shape(centroids_0) != np.shape(centroids_1):
            #find what the difference in shape is
            missing_rows = np.shape(centroids_0)[0] - np.shape(centroids_1)[0]
            pad = np.ones((abs(missing_rows),4)) * -1000
            pad[:,-1] = -1
            if missing_rows > 0:
                centroids_1 = np.concatenate((centroids_1, pad), axis=0)
            else:
                centroids_0 = np.concatenate((centroids_0, pad), axis=0)
        distances = cdist(centroids_0[:,:-1], centroids_1[:,:-1]) #find distance using xyz but not label
        closest_indices = np.argmin(distances, axis=1)
        pairs = np.concatenate((centroids_1, centroids_0[closest_indices]), axis=1)
    
        #WARN: matcher indices will break if format changes from x,y,z, label
        if missing_rows == 0:
            #if all labels in centroids_0 (previous frame) have match in centroids_1(current frame)
            matcher = {i[3] : i[7] for i in pairs}
        # Ver1
        # elif missing_rows < 0:  
        # #if there are fewer labels in previous frame than this frame
        #     #for any values in this frame that are above the index of last frame, leave them as is (dict = to self)
        #     matcher = {i[3] : (i[7] if (i[3] >=0 and i[7] >=0) else i[3]) for i in pairs}
        # else:  
        # #if there are more labels in previous frame than this frame: 
        #     #the labels shouldn't get used, but leave them as is (set to self)   
        #     matcher = {(i[3] if (i[3] >=0 and i[7] >=0) else i[7]) : i[7] for i in pairs}

        ## ver 2.01
        else: 
        #trying to fix my decay to 1 or 2 colors problem
            matcher = {i[3] : i[7] for i in pairs if (i[3] >=0 and i[7]) >=0 }

    #    # #ver1
    #     matched_labels = np.array([int(matcher[i]) for i in AHC_model_labels])
    #     self.get_logger().info("matched_labels shape %s" %np.shape(matched_labels))

        # ver2 
        # for i in np.arange(AHC_model_labels.shape[0]):
        #     try: #try to match value
        #         matched_labels[i] = int(matcher[AHC_model_labels[i]]) 
        #     except: #if we can't match value, that means it isn't in matcher. Then just use the label value
        #         matched_labels[i] = AHC_model_labels[i]

        #ver 3.01  #lookup the key = i in dict and return value, else use i itself. 
        # breakpoint()
        matched_labels = np.array([matcher.get(i, i) for i in AHC_model_labels])
        self.get_logger().info("matched_labels shape %s" %np.shape(matched_labels))
        ## saving the new labels to centroid1 so we can compare in the next frame
        # centroids_1[:,-1] = [matcher[i] for i in np.arange(centroids_1.shape[0])]
        for i in np.arange(centroids_1.shape[0]):
            try:
                centroids_1[i,-1] = matcher[i]
            except: 
                # print("invalid key? i = ",i)
                centroids_1[i,-1] = centroids_1.shape[0]+1
        print('matcher ------', matcher)
        #TODO if anything is still -1000 at the end I'll set it equal to its key before the translation
        #TODO check the size of AHC_model_labels before manipulating and check again after matched_labels
        self.get_logger().debug("finished match_labelnames")
        return self, matched_labels, centroids_1
        #try mapping the label, except leave it as is without changing dict, -1000 = max + 1? 

    #reads rosparam and automatically adjusts downsample to yield desired framerate. 
    def control_framerate(self, len_x):
        if self.frame_time <= self.AHC_period_:
            self.get_logger().debug("difference between goal time and actual time: %.4f" % (self.AHC_period_ - self.frame_time))
        else:     # when self.frame_time > self.AHC_period_
            self.get_logger().debug("too long: difference between goal time and actual time: %.4f" % (self.AHC_period_ - self.frame_time))
        #time complexity of AHC is  O(N^3)
        # equation: time constant = current period / current points ^3
        time_constant = self.frame_time / (len_x **3) #1 in notebook
        # equation: goal number of points = (goal time/time_constant)^(1/3)
        goal_points = (self.AHC_period_/ time_constant)**(1/3) #2 in notebook
        # equation: new downsample number = current points* current downsample / goal points
        self.downsample = int((len_x*self.downsample) / goal_points) + 1  #3 in notebook. Int rounds down, then +1, is easier than importing math for ceil. 
        self.get_logger().debug("+ time_constant = %.5s, goal_points = %.5f, downsample = %s" %(time_constant, goal_points, self.downsample))
        return self


    def listener_callback(self, pc2_msg):
    ### minimize
        # self.get_logger().debug("Received pcl2")
        start = timeit.default_timer()

        ## color map options for plots (not using, rviz instead)
        # c1 = cm.gist_rainbow    #c2 = cm.plasma_r
        # c3 = cm.rainbow     # c4 = cm.jet
        #for debugging lidar pcl2 msg
        # fields = pc2_msg.fields  
        # self.get_logger().info('fields.type %s' % (type(fields))) 

        ##get xyz data from pcl2 msg 
        ##added np.array to convert list to array
        self.xyz = tuple(map(tuple,(list(point_cloud2.read_points(pc2_msg, field_names=('x','y','z',))))))
        self.xyz = list(self.xyz)
        #breakpoint()
        # ii = list(point_cloud2.read_points(pc2_msg, field_names=('intensity')))  #comment to improve processing time
        # rr = list(point_cloud2.read_points(pc2_msg, field_names=('ring')))        #comment to improve processing time
        # self.get_logger().info('read points time %.7s' %(timeit.default_timer() - start) )

        ## obsolete when using rviz
        # show original xyz
        # title = "Original xyz"
        # self.plot_3d_scatter(x,y,z,c=x,cmap=c1, norm_color=4.2, figsize=(18,12),show=-1, title=title)

        # title = "xyz 10x downsampled"
        self = self.decimate_pcl2(step_size = self.downsample)
        self, x,y,z= self.format_pcl2()

        ## do clustering, find ACH
        self.get_logger().debug('~time before clustering %.7s' %(timeit.default_timer() - start) )
        # start = timeit.default_timer()

        d = self.AHC_distance_ 

    ### for operating
        # try:
        #     # breakpoint()
        #     model = AgglomerativeClustering(distance_threshold=d, n_clusters=None)
        #     model = model.fit( np.array((x,y,z)).T ) 
        #     self.get_logger().info('AHC distance_threshold: %s -- clusters: %s' % (d, max(model.labels_)))
        #     # startc = timeit.default_timer()
        #     self, centroid1 = self.get_centroids( model.labels_)
        #     self.create_pcl2_and_pub(x,y,z, model.labels_)
        #     # self.centroid0 = centroid1  
        #     # self.get_logger().info('.cluster label time %.7s' %(timeit.default_timer() - startc) )
        #     self.frame_time = round(timeit.default_timer() - start, 3)
        #     self.get_logger().warn('  frame time %s' %self.frame_time)
        #     self.control_framerate(len(x))     
        # except:
        #     self.get_logger().error("would've crashed")

    ### for debugging
        # breakpoint()
        model = AgglomerativeClustering(distance_threshold=d, n_clusters=None)
        model = model.fit( np.array((x,y,z)).T ) 
        self.get_logger().info('AHC distance_threshold: %s -- clusters: %s' % (d, max(model.labels_)+1))
        # startc = timeit.default_timer()
        self, centroids1 = self.get_centroids(model.labels_)
        self.create_pcl2_and_pub(x,y,z, model.labels_)
        # self.centroid0 = centroid1  
        # self.get_logger().info('.cluster label time %.7s' %(timeit.default_timer() - startc) )
        self.frame_time = round(timeit.default_timer() - start, 3)
        self.get_logger().warn('  Lidar clustering: frame time %s --- fps %.1f' %(self.frame_time, 1/self.frame_time))
        self.control_framerate(len(x))     

        self.pub_centroids(centroids1)
        # self, bounding_boxes = self.get_pub_bounding_boxes(model.labels_)

    ### TODO keep consistent cluster labels from frame to frame
        # startc = timeit.default_timer()
        # self, centroid1 = self.get_centroids(model.labels_)
        # breakpoint()
        #consistent labels not working great at the moment. 
        # if self.first:
        #     self, matched_labels, centroid1= self.match_labelnames(self.centroid0, centroid1, model.labels_)
        #     self.create_pcl2_and_pub(x,y,z, matched_labels)

        # labels as outputted
        # self.create_pcl2_and_pub(x,y,z, model.labels_)

        # self.centroid0 = centroid1  
        # self.get_logger().info('.cluster label time %.7s' %(timeit.default_timer() - startc) )
        # self.get_logger().warn('~~~frame time %.7s' %(timeit.default_timer() - start) )

        ## plot the top p levels of the dendrogram
        # self.plot_dendrogram(model, truncate_mode="level", p=4)
        # plt.show(block=False)
        # plt.pause(0.5)
        # plt.close()
        ## show scatteplot of detections
        # title = ("detections at d = %s" %d)
        # self.plot_3d_scatter(x,y,z,c=model.labels_, cmap=c2, norm_color='labels',figsize=(18,12),show=-1,show_surface=1,cbar=0, title=title)

    # to help get consistent labels
        self.first = 0


def main(args=None):
    rclpy.init(args=args)
    AHC_node = pcl2Subscriber()
    rclpy.spin(AHC_node)
    AHC_node.destroy_node() 
    rclpy.shutdown()

if __name__ == '__main__':
    main()