import numpy as np


class BodyJoints26FK:
    """
    Joint type defined for easier forward kinematic using
    """
    def __init__(self):

        # name
        self.pose_type = "body26fk"

        # number of joints
        self.count = 26

        # indexes of joints
        self.pelvis = 0
        self.left_hip = 1
        self.right_hip = 2
        self.torso = 3
        self.left_knee = 4
        self.right_knee = 5
        self.neck = 6
        self.left_ankle = 7
        self.right_ankle = 8
        self.left_big_toe = 9
        self.right_big_toe = 10
        self.left_small_toe = 11
        self.right_small_toe = 12
        self.left_heel = 13
        self.right_heel = 14
        self.nose = 15
        self.left_eye = 16
        self.right_eye = 17
        self.left_ear = 18
        self.right_ear = 19
        self.left_shoulder = 20
        self.right_shoulder = 21
        self.left_elbow = 22
        self.right_elbow = 23
        self.left_wrist = 24
        self.right_wrist = 25
        # Use wrist as root joint
        self.root = self.pelvis

        # Joint names
        self.name = dict()
        self.name[self.right_ankle]    = "right_ankle"
        self.name[self.right_knee]     = "right_knee"
        self.name[self.right_hip]      = "right_hip"
        self.name[self.right_shoulder] = "right_shoulder"
        self.name[self.right_elbow]    = "right_elbow"
        self.name[self.right_wrist]    = "right_wrist"
        self.name[self.left_ankle]     = "left_ankle"
        self.name[self.left_knee]      = "left_knee"
        self.name[self.left_hip]       = "left_hip"
        self.name[self.left_shoulder]  = "left_shoulder"
        self.name[self.left_elbow]     = "left_elbow"
        self.name[self.left_wrist]     = "left_wrist"
        self.name[self.neck]           = "neck"
        self.name[self.pelvis]         = "pelvis"
        self.name[self.torso]          = "torso"
        self.name[self.nose]           = "nose"
        self.name[self.right_eye]      = "right_eye"
        self.name[self.right_ear]      = "right_ear"
        self.name[self.left_eye] = "left_eye"
        self.name[self.left_ear] = "left_ear"
        self.name[self.right_heel] = "right_heel"
        self.name[self.right_big_toe] = "right_big_toe"
        self.name[self.right_small_toe] = "right_small_toe"
        self.name[self.left_heel] = "left_heel"
        self.name[self.left_big_toe] = "left_big_toe"
        self.name[self.left_small_toe] = "left_small_toe"

        # Edges
        self.edges = [[self.pelvis, self.torso], # 0
                      [self.torso, self.neck], # 1
                      [self.neck, self.pelvis],  # 2
                      [self.right_ankle, self.right_knee], # 3
                      [self.right_knee, self.right_hip], # 4
                      [self.right_hip, self.pelvis], # 5
                      [self.right_hip, self.right_shoulder],  # 6
                      [self.right_shoulder, self.right_elbow], # 7
                      [self.right_elbow, self.right_wrist], # 8
                      [self.left_ankle, self.left_knee], # 9
                      [self.left_knee, self.left_hip], # 10
                      [self.left_hip, self.pelvis], # 11
                      [self.left_hip, self.left_shoulder], # 12
                      [self.left_shoulder, self.left_elbow], # 13
                      [self.left_elbow, self.left_wrist], # 14
                      [self.right_shoulder, self.neck], # 15
                      [self.left_shoulder, self.neck], # 16
                      # [self.neck, self.head], # 17
                      [self.neck, self.nose], # 18
                      # [self.nose, self.head],  # 19
                      [self.nose, self.right_eye], # 20
                      [self.right_eye, self.right_ear], # 21
                      [self.nose, self.left_eye], # 22
                      [self.left_eye, self.left_ear], # 23
                      [self.right_ankle, self.right_heel],
                      [self.right_ankle, self.right_big_toe],
                      [self.right_big_toe, self.right_small_toe],
                      [self.left_ankle, self.left_heel],
                      [self.left_ankle, self.left_big_toe],
                      [self.left_big_toe, self.left_small_toe]]


        # symmetric edges
        self.symmetric_edges = [([self.right_ankle, self.right_knee], [self.left_ankle, self.left_knee]),
                                ([self.right_knee, self.right_hip], [self.left_knee, self.left_hip]),
                                ([self.right_hip, self.pelvis], [self.left_hip, self.pelvis]),
                                ([self.right_shoulder, self.right_elbow], [self.left_shoulder, self.left_elbow]),
                                ([self.right_elbow, self.right_wrist], [self.left_elbow, self.left_wrist]),
                                ([self.right_shoulder, self.neck], [self.left_shoulder, self.neck]),
                                ([self.nose, self.right_eye], [self.nose, self.left_eye]),
                                ([self.right_eye, self.right_ear], [self.left_eye, self.left_ear]),
                                ([self.right_ankle, self.right_heel], [self.left_ankle, self.left_heel]),
                                ([self.right_ankle, self.right_big_toe], [self.left_ankle, self.left_big_toe]),
                                ([self.right_ankle, self.right_small_toe], [self.left_ankle, self.left_small_toe])]

        # colors for for each edge
        self.edge_colors = np.array([
                            [0, 1., 0.12745098],
                            [0, 1., 0.12745098],
                            [0, 1., 0.12745098],
                            [0., 0.12745098, 1.],
                            [0., 0.12745098, 1.],
                            [0, 1., 0.12745098],
                            [0., 0.12745098, 1.],
                            [0., 0.12745098, 1.],
                            [0., 0.12745098, 1.],
                            [1., 0.12745098, 0],
                            [1., 0.12745098, 0],
                            [0, 1., 0.12745098],
                            [1., 0.12745098, 0],
                            [1., 0.12745098, 0],
                            [1., 0.12745098, 0],
                            [0, 1., 0.12745098],
                            [0, 1., 0.12745098],
                            [0, 1., 0.12745098],
                            [0, 1., 0.12745098],
                            [0, 1., 0.12745098],
                            [0, 1., 0.12745098],
                            [0, 1., 0.12745098],
                            [0., 0.12745098, 1.],
                            [0., 0.12745098, 1.],
                            [0., 0.12745098, 1.],
                            [1., 0.12745098, 0],
                            [1., 0.12745098, 0],
                            [1., 0.12745098, 0]])

        self.symmetric_joint = dict()
        self.symmetric_joint[self.right_ankle]    = self.left_ankle
        self.symmetric_joint[self.right_knee]     = self.left_knee
        self.symmetric_joint[self.right_hip]      = self.left_hip
        self.symmetric_joint[self.right_shoulder] = self.left_shoulder
        self.symmetric_joint[self.right_elbow]    = self.left_elbow
        self.symmetric_joint[self.right_wrist]    = self.left_wrist
        self.symmetric_joint[self.left_ankle]     = self.right_ankle
        self.symmetric_joint[self.left_knee]      = self.right_knee
        self.symmetric_joint[self.left_hip]       = self.right_hip
        self.symmetric_joint[self.left_shoulder]  = self.right_shoulder
        self.symmetric_joint[self.left_elbow]     = self.right_elbow
        self.symmetric_joint[self.left_wrist]     = self.right_wrist
        self.symmetric_joint[self.neck]           = -1
        self.symmetric_joint[self.torso]          = -1
        self.symmetric_joint[self.pelvis]         = -1
        self.symmetric_joint[self.nose]           = -1
        self.symmetric_joint[self.right_eye]      = self.left_eye
        self.symmetric_joint[self.right_ear]      = self.left_ear
        self.symmetric_joint[self.left_ear]       = self.right_ear
        self.symmetric_joint[self.left_eye]       = self.right_eye
        self.symmetric_joint[self.right_heel] = self.left_heel
        self.symmetric_joint[self.right_big_toe] = self.left_big_toe
        self.symmetric_joint[self.right_small_toe] = self.left_small_toe
        self.symmetric_joint[self.left_heel]      = self.right_heel
        self.symmetric_joint[self.left_big_toe]   = self.right_big_toe
        self.symmetric_joint[self.left_small_toe] = self.right_small_toe

        self.sigmas = np.array([1.07, 1.07, 0.89, 0.87, 1.07, 1.07, 0.87, 0.89, 0.62, 0.72, 0.79, 0.79, 0.72, 0.62, 0.79, 0.79, 0.26, 0.25, 0.35, 0.25, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35]) / 10.0

                        # parent,       #child
        self.parents = [-1, #pelvis = 0
                        self.pelvis,    #left_hip = 1
                        self.pelvis,    #right_hip = 2
                        self.pelvis,    #torso = 3
                        self.left_hip,  #left_knee = 4
                        self.right_hip, #right_knee = 5
                        self.torso,         #neck = 6
                        self.left_knee,     #left_ankle = 7
                        self.right_knee,    #right_ankle = 8
                        self.left_ankle,    #left_big_toe = 9
                        self.right_ankle,   #right_big_toe = 10
                        self.left_ankle,    #left_small_toe = 11
                        self.right_ankle,   #right_small_toe = 12
                        self.left_ankle,    #left_heel = 13
                        self.right_ankle,   #right_heel = 14
                        self.neck,          #nose = 15
                        self.nose,          #left_eye = 16
                        self.nose,          #right_eye = 17
                        self.nose,          #left_ear = 18
                        self.nose,          #right_ear = 19
                        self.neck,          #left_shoulder = 20
                        self.neck,          #right_shoulder = 21
                        self.left_shoulder, #left_elbow = 22
                        self.right_shoulder, #right_elbow = 23
                        self.left_elbow,     #left_wrist = 24
                        self.right_elbow]    #right_wrist = 25


class BodyJoints30:
    """
    Joint type defined for easier forward kinematic using
    """
    def __init__(self):

        # name
        self.pose_type = "body30"

        # number of joints
        self.count = 30

        # indexes of joints
        self.pelvis = 0
        self.left_hip = 1
        self.right_hip = 2
        self.torso = 3
        self.left_knee = 4
        self.right_knee = 5
        self.neck = 6
        self.left_ankle = 7
        self.right_ankle = 8
        self.left_big_toe = 9
        self.right_big_toe = 10
        self.left_small_toe = 11
        self.right_small_toe = 12
        self.left_heel = 13
        self.right_heel = 14
        self.nose = 15
        self.left_eye = 16
        self.right_eye = 17
        self.left_ear = 18
        self.right_ear = 19
        self.left_shoulder = 20
        self.right_shoulder = 21
        self.left_elbow = 22
        self.right_elbow = 23
        self.left_wrist = 24
        self.right_wrist = 25
        self.left_pinky_knuckle = 26
        self.right_pinky_knuckle = 27
        self.left_index_knuckle = 28
        self.right_index_knuckle = 29

        # Use wrist as root joint
        self.root = self.pelvis

        # Joint names
        self.name = dict()
        self.name[self.right_ankle]    = "right_ankle"
        self.name[self.right_knee]     = "right_knee"
        self.name[self.right_hip]      = "right_hip"
        self.name[self.right_shoulder] = "right_shoulder"
        self.name[self.right_elbow]    = "right_elbow"
        self.name[self.right_wrist]    = "right_wrist"
        self.name[self.right_pinky_knuckle]    = "right_pinky_knuckle"
        self.name[self.right_index_knuckle]    = "right_index_knuckle"
        self.name[self.left_ankle]     = "left_ankle"
        self.name[self.left_knee]      = "left_knee"
        self.name[self.left_hip]       = "left_hip"
        self.name[self.left_shoulder]  = "left_shoulder"
        self.name[self.left_elbow]     = "left_elbow"
        self.name[self.left_wrist]     = "left_wrist"
        self.name[self.left_pinky_knuckle] = "left_pinky_knuckle"
        self.name[self.left_index_knuckle] = "left_index_knuckle"
        self.name[self.neck]           = "neck"
        self.name[self.pelvis]         = "pelvis"
        self.name[self.torso]          = "torso"
        self.name[self.nose]           = "nose"
        self.name[self.right_eye]      = "right_eye"
        self.name[self.right_ear]      = "right_ear"
        self.name[self.left_eye] = "left_eye"
        self.name[self.left_ear] = "left_ear"
        self.name[self.right_heel] = "right_heel"
        self.name[self.right_big_toe] = "right_big_toe"
        self.name[self.right_small_toe] = "right_small_toe"
        self.name[self.left_heel] = "left_heel"
        self.name[self.left_big_toe] = "left_big_toe"
        self.name[self.left_small_toe] = "left_small_toe"

        # Edges
        self.edges = [[self.pelvis, self.torso], # 0
                      [self.torso, self.neck], # 1
                      [self.neck, self.pelvis],  # 2

                      [self.right_ankle, self.right_knee], # 3
                      [self.right_knee, self.right_hip], # 4
                      [self.right_hip, self.pelvis], # 5
                      [self.right_hip, self.right_shoulder],  # 6
                      [self.right_shoulder, self.right_elbow], # 7
                      [self.right_elbow, self.right_wrist], # 8
                      [self.right_wrist, self.right_pinky_knuckle],
                      [self.right_wrist, self.right_index_knuckle],
                      [self.right_pinky_knuckle, self.right_index_knuckle],
                      [self.right_ankle, self.right_heel],
                      [self.right_ankle, self.right_big_toe],
                      [self.right_big_toe, self.right_small_toe],
                      [self.right_shoulder, self.neck], # 15

                      [self.left_ankle, self.left_knee], # 9
                      [self.left_knee, self.left_hip], # 10
                      [self.left_hip, self.pelvis], # 11
                      [self.left_hip, self.left_shoulder], # 12
                      [self.left_shoulder, self.left_elbow], # 13
                      [self.left_elbow, self.left_wrist], # 14
                      [self.left_wrist, self.left_pinky_knuckle],
                      [self.left_wrist, self.left_index_knuckle],
                      [self.left_pinky_knuckle, self.left_index_knuckle],
                      [self.left_ankle, self.left_heel],
                      [self.left_ankle, self.left_big_toe],
                      [self.left_big_toe, self.left_small_toe],
                      [self.left_shoulder, self.neck], # 16

                      [self.neck, self.nose], # 18
                      [self.nose, self.right_eye], # 20
                      [self.right_eye, self.right_ear], # 21
                      [self.nose, self.left_eye], # 22
                      [self.left_eye, self.left_ear]]


        # symmetric edges
        self.symmetric_edges = [([self.right_ankle, self.right_knee], [self.left_ankle, self.left_knee]),
                                ([self.right_knee, self.right_hip], [self.left_knee, self.left_hip]),
                                ([self.right_hip, self.pelvis], [self.left_hip, self.pelvis]),
                                ([self.right_shoulder, self.right_elbow], [self.left_shoulder, self.left_elbow]),
                                ([self.right_elbow, self.right_wrist], [self.left_elbow, self.left_wrist]),
                                ([self.right_wrist, self.right_pinky_knuckle], [self.left_wrist, self.left_pinky_knuckle]),
                                ([self.right_wrist, self.right_index_knuckle], [self.left_wrist, self.left_index_knuckle]),
                                ([self.right_shoulder, self.neck], [self.left_shoulder, self.neck]),
                                ([self.nose, self.right_eye], [self.nose, self.left_eye]),
                                ([self.right_eye, self.right_ear], [self.left_eye, self.left_ear]),
                                ([self.right_ankle, self.right_heel], [self.left_ankle, self.left_heel]),
                                ([self.right_ankle, self.right_big_toe], [self.left_ankle, self.left_big_toe]),
                                ([self.right_ankle, self.right_small_toe], [self.left_ankle, self.left_small_toe])]

                                # colors for for each edge
        self.edge_colors = np.array([
                            [0, 1., 0.12745098],
                            [0, 1., 0.12745098],
                            [0, 1., 0.12745098],

                            [0., 0.12745098, 1.],
                            [0., 0.12745098, 1.],
                            [0., 0.12745098, 1.],
                            [0., 0.12745098, 1.],
                            [0., 0.12745098, 1.],
                            [0., 0.12745098, 1.],
                            [0., 0.12745098, 1.],
                            [0., 0.12745098, 1.],
                            [0., 0.12745098, 1.],
                            [0., 0.12745098, 1.],
                            [0., 0.12745098, 1.],
                            [0., 0.12745098, 1.],
                            [0., 0.12745098, 1.],

                            [1., 0.12745098, 0],
                            [1., 0.12745098, 0],
                            [1., 0.12745098, 0],
                            [1., 0.12745098, 0],
                            [1., 0.12745098, 0],
                            [1., 0.12745098, 0],
                            [1., 0.12745098, 0],
                            [1., 0.12745098, 0],
                            [1., 0.12745098, 0],
                            [1., 0.12745098, 0],
                            [1., 0.12745098, 0],
                            [1., 0.12745098, 0],
                            [1., 0.12745098, 0],

                            [0, 1., 0.12745098],
                            [0, 1., 0.12745098],
                            [0, 1., 0.12745098],
                            [0, 1., 0.12745098],
                            [0, 1., 0.12745098]])


        self.symmetric_joint = dict()
        self.symmetric_joint[self.right_ankle]    = self.left_ankle
        self.symmetric_joint[self.right_knee]     = self.left_knee
        self.symmetric_joint[self.right_hip]      = self.left_hip
        self.symmetric_joint[self.right_shoulder] = self.left_shoulder
        self.symmetric_joint[self.right_elbow]    = self.left_elbow
        self.symmetric_joint[self.right_wrist]    = self.left_wrist
        self.symmetric_joint[self.right_pinky_knuckle]  = self.left_pinky_knuckle
        self.symmetric_joint[self.right_index_knuckle]  = self.left_index_knuckle
        self.symmetric_joint[self.left_ankle]     = self.right_ankle
        self.symmetric_joint[self.left_knee]      = self.right_knee
        self.symmetric_joint[self.left_hip]       = self.right_hip
        self.symmetric_joint[self.left_shoulder]  = self.right_shoulder
        self.symmetric_joint[self.left_elbow]     = self.right_elbow
        self.symmetric_joint[self.left_wrist]     = self.right_wrist
        self.symmetric_joint[self.left_pinky_knuckle] = self.right_pinky_knuckle
        self.symmetric_joint[self.left_index_knuckle] = self.right_index_knuckle
        self.symmetric_joint[self.neck]           = -1
        self.symmetric_joint[self.torso]          = -1
        self.symmetric_joint[self.pelvis]         = -1
        self.symmetric_joint[self.nose]           = -1
        self.symmetric_joint[self.right_eye]      = self.left_eye
        self.symmetric_joint[self.right_ear]      = self.left_ear
        self.symmetric_joint[self.left_ear]       = self.right_ear
        self.symmetric_joint[self.left_eye]       = self.right_eye
        self.symmetric_joint[self.right_heel]     = self.left_heel
        self.symmetric_joint[self.right_big_toe]  = self.left_big_toe
        self.symmetric_joint[self.right_small_toe] = self.left_small_toe
        self.symmetric_joint[self.left_heel]      = self.right_heel
        self.symmetric_joint[self.left_big_toe]   = self.right_big_toe
        self.symmetric_joint[self.left_small_toe] = self.right_small_toe

        self.sigmas = np.array([1.07, 1.07, 0.89, 0.87, 1.07, 1.07, 0.87, 0.89, 0.62, 0.72, 0.79, 0.79, 0.72, 0.62, 0.79, 0.79, 0.26, 0.25, 0.35, 0.25, 0.35]) / 10.0

                        # parent,       #child
        self.parents = [-1, #pelvis = 0
                        self.pelvis,    #left_hip = 1
                        self.pelvis,    #right_hip = 2
                        self.pelvis,    #torso = 3
                        self.left_hip,  #left_knee = 4
                        self.right_hip, #right_knee = 5
                        self.torso,         #neck = 6
                        self.left_knee,     #left_ankle = 7
                        self.right_knee,    #right_ankle = 8
                        self.left_ankle,    #left_big_toe = 9
                        self.right_ankle,   #right_big_toe = 10
                        self.left_ankle,    #left_small_toe = 11
                        self.right_ankle,   #right_small_toe = 12
                        self.left_ankle,    #left_heel = 13
                        self.right_ankle,   #right_heel = 14
                        self.neck,          #nose = 15
                        self.nose,          #left_eye = 16
                        self.nose,          #right_eye = 17
                        self.nose,          #left_ear = 18
                        self.nose,          #right_ear = 19
                        self.neck,          #left_shoulder = 20
                        self.neck,          #right_shoulder = 21
                        self.left_shoulder, #left_elbow = 22
                        self.right_shoulder, #right_elbow = 23
                        self.left_elbow,     #left_wrist = 24
                        self.right_elbow,    #right_wrist = 25
                        self.left_wrist,     #left_pinky_knuckle = 26
                        self.right_wrist,  #right_pinky_knuckle = 27
                        self.left_wrist,   #left_index_knuckle = 28
                        self.right_wrist  #right_index_knuckle = 29
                        ]


class BodyJointsCOCO:
    def __init__(self):

        # name
        self.pose_type = "coco"

        # number of joints
        self.count      = 18

        # indexes of joints
        self.nose = 0
        self.neck = 1
        self.right_shoulder = 2
        self.right_elbow = 3
        self.right_wrist = 4
        self.left_shoulder = 5
        self.left_elbow = 6
        self.left_wrist = 7
        self.right_hip = 8
        self.right_knee = 9
        self.right_ankle = 10
        self.left_hip = 11
        self.left_knee = 12
        self.left_ankle = 13
        self.right_eye = 14
        self.left_eye = 15
        self.right_ear = 16
        self.left_ear = 17
        self.root = self.neck # just use whatever

        # Joint names
        self.name = dict()
        self.name[self.right_ankle]    = "right_ankle"
        self.name[self.right_knee]     = "right_knee"
        self.name[self.right_hip]      = "right_hip"
        self.name[self.right_shoulder] = "right_shoulder"
        self.name[self.right_elbow]    = "right_elbow"
        self.name[self.right_wrist]    = "right_wrist"
        self.name[self.left_ankle]     = "left_ankle"
        self.name[self.left_knee]      = "left_knee"
        self.name[self.left_hip]       = "left_hip"
        self.name[self.left_shoulder]  = "left_shoulder"
        self.name[self.left_elbow]     = "left_elbow"
        self.name[self.left_wrist]     = "left_wrist"
        self.name[self.nose]           = "nose"
        self.name[self.right_eye]      = "right_eye"
        self.name[self.right_ear]      = "right_ear"
        self.name[self.left_eye] = "left_eye"
        self.name[self.left_ear] = "left_ear"
        self.name[self.neck] = "neck"

        # Edges
        self.edges = [[self.right_ankle, self.right_knee], # 3
                      [self.right_knee, self.right_hip], # 4
                      [self.right_hip, self.right_shoulder],  # 6
                      [self.right_shoulder, self.right_elbow], # 7
                      [self.right_elbow, self.right_wrist], # 8
                      [self.left_ankle, self.left_knee], # 9
                      [self.left_knee, self.left_hip], # 10
                      [self.left_hip, self.left_shoulder], # 12
                      [self.left_shoulder, self.left_elbow], # 13
                      [self.left_elbow, self.left_wrist], # 14
                      [self.right_shoulder, self.neck], # 15
                      [self.left_shoulder, self.neck], # 16
                      [self.neck, self.nose], # 18
                      [self.nose, self.right_eye], # 20
                      [self.right_eye, self.right_ear], # 21
                      [self.nose, self.left_eye], # 22
                      [self.left_eye, self.left_ear]]


        # symmetric edges
        self.symmetric_edges = [([self.right_ankle, self.right_knee], [self.left_ankle, self.left_knee]),
                                ([self.right_knee, self.right_hip], [self.left_knee, self.left_hip]),
                                ([self.right_shoulder, self.right_elbow], [self.left_shoulder, self.left_elbow]),
                                ([self.right_elbow, self.right_wrist], [self.left_elbow, self.left_wrist])]

        # colors for for each edge
        self.edge_colors = np.array([
                            [0, 1., 0.12745098],
                            [0, 1., 0.12745098],
                            [0., 0.12745098, 1.],
                            [0., 0.12745098, 1.],
                            [0., 0.12745098, 1.],
                            [0., 0.12745098, 1.],
                            [0., 0.12745098, 1.],
                            [1., 0.12745098, 0],
                            [1., 0.12745098, 0],
                            [1., 0.12745098, 0],
                            [1., 0.12745098, 0],
                            [1., 0.12745098, 0],
                            [0., 0.12745098, 1.],
                            [1., 0.12745098, 0],
                            [0, 1., 0.12745098],
                            [0, 1., 0.12745098],
                            [0, 1., 0.12745098],
                            [0, 1., 0.12745098],
                            [0, 1., 0.12745098]])

        self.symmetric_joint = dict()
        self.symmetric_joint[self.right_ankle]    = self.left_ankle
        self.symmetric_joint[self.right_knee]     = self.left_knee
        self.symmetric_joint[self.right_hip]      = self.left_hip
        self.symmetric_joint[self.right_shoulder] = self.left_shoulder
        self.symmetric_joint[self.right_elbow]    = self.left_elbow
        self.symmetric_joint[self.right_wrist]    = self.left_wrist
        self.symmetric_joint[self.left_ankle]     = self.right_ankle
        self.symmetric_joint[self.left_knee]      = self.right_knee
        self.symmetric_joint[self.left_hip]       = self.right_hip
        self.symmetric_joint[self.left_shoulder]  = self.right_shoulder
        self.symmetric_joint[self.left_elbow]     = self.right_elbow
        self.symmetric_joint[self.left_wrist]     = self.right_wrist
        self.symmetric_joint[self.nose]           = -1
        self.symmetric_joint[self.right_eye]      = self.left_eye
        self.symmetric_joint[self.right_ear]      = self.left_ear
        self.symmetric_joint[self.left_ear]       = self.right_ear
        self.symmetric_joint[self.left_eye]       = self.right_eye

        self.sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0


class SMPLJoints:
    """
    Joint type defined for easier forward kinematic using
    """
    def __init__(self):

        # name
        self.pose_type = "smpl"

        # number of joints
        self.count = 24

        # indexes of joints
        self.pelvis = 0
        self.left_hip = 1
        self.right_hip = 2
        self.spine1 = 3
        self.left_knee = 4
        self.right_knee = 5
        self.spine2 = 6
        self.left_ankle = 7
        self.right_ankle = 8
        self.spine3 = 9
        self.left_foot = 10
        self.right_foot = 11
        self.neck = 12
        self.left_collar = 13
        self.right_collar = 14
        self.head = 15
        self.left_shoulder = 16
        self.right_shoulder = 17
        self.left_elbow = 18
        self.right_elbow = 19
        self.left_wrist = 20
        self.right_wrist = 21
        self.left_hand = 22
        self.right_hand = 23

        # Use wrist as root joint
        self.root = self.pelvis

        # Joint names
        self.name = dict()
        self.name[self.right_ankle]    = "right_ankle"
        self.name[self.right_knee]     = "right_knee"
        self.name[self.right_hip]      = "right_hip"
        self.name[self.right_shoulder] = "right_shoulder"
        self.name[self.right_elbow]    = "right_elbow"
        self.name[self.right_wrist]    = "right_wrist"
        self.name[self.left_ankle]     = "left_ankle"
        self.name[self.left_knee]      = "left_knee"
        self.name[self.left_hip]       = "left_hip"
        self.name[self.left_shoulder]  = "left_shoulder"
        self.name[self.left_elbow]     = "left_elbow"
        self.name[self.left_wrist]     = "left_wrist"
        self.name[self.neck]           = "neck"
        self.name[self.pelvis]         = "pelvis"
        self.name[self.spine1]         = "spine1"
        self.name[self.spine2]         = "spine2"
        self.name[self.spine3]         = "spine3"
        self.name[self.left_collar]    = "left_collar"
        self.name[self.right_collar]   = "right_collar"
        self.name[self.left_hand]      = "left_hand"
        self.name[self.right_hand]     = "right_hand"
        self.name[self.left_foot]     = "left_foot"
        self.name[self.right_foot] = "right_foot"


def get_joints_info(pose_type):

    if pose_type == "body26fk":
        return BodyJoints26FK()
    elif pose_type == "body30":
        return BodyJoints30()
    elif pose_type == "coco":
        return BodyJointsCOCO()
    elif pose_type == "smpl":
        return SMPLJoints()
    else:
       raise ValueError("Unknown body_type provided.")
