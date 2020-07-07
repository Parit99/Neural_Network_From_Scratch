 output_list=[]
class Convnet:
    np.random.seed(0)
   
    def __init__(self,inputs,n_neurons=1):
        self.weights_list=[]
        self.biases_list=[]
        self.output_list=[]
        self.activation_error=[]
        self.inputs=inputs
        n_inputs=inputs.shape[1]
        self.weights=np.random.randn(n_inputs,n_neurons)
        self.weights_list.append(self.weights)
        self.biases=np.zeros((1,n_neurons))
        self.biases_list.append(self.biases)
        print('Number of weights :: ',self.weights.shape)
        
    def forward(self,label,activation):
        self.activation=activation
        self.label=label
        self.output=np.dot(self.inputs,self.weights)+self.biases
        output_list.append([self.output,self.label])
        if(self.activation=='relu'):
            relu=Activation_relu(self.output)
            rf=relu.forward()
            self.activation_error.append(rf)
            return rf
        if(self.activation=='sigmoid'):
            sigmoid=Activation_sigmoid(self.output)
            sf=sigmoid.forward()
            self.activation_error.append(sf)
            return sf
    
    def Conv2D(self,kernel,X,activation):
        self.img=X
        self.y=y
        self.activation=activation
        filter_horedge   = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        filter_veredge   = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        filter_edge      = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
        filter_diagonal  = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
        filter_sharpen   = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        filter_gaussblur = np.array([[1,2,1],[2,4,2],[1,2,1]])
        filter_gaussblur =1/16*filter_gaussblur
        filter_smooth    =np.array([[1,1,1],[1,1,1],[1,1,1]])
        filter_smooth    =1/9*filter_smooth
        if(kernel==3):
            op           =cv2.filter2D(X,-1,filter_gaussblur)
            op1          =cv2.filter2D(op,-1,filter_smooth)
            op2          =cv2.filter2D(op1,-1,filter_sharpen)
            op3          =cv2.filter2D(op2,-1,filter_horedge)
            self.output  =cv2.filter2D(op3,-1,filter_veredge)
            clip         = np.floor(np.array(filter_veredge.shape)/2).astype(np.int) # Find half dims of kernel
            self.output  = self.output[clip[0]:-clip[0],clip[1]:-clip[1]]
            if(self.activation=='relu'):
                relu=Activation_relu(self.output)
                rf=relu.forward()
                #print(rf)
                return rf
            if(self.activation=='sigmoid'):
                sigmoid=Activation_sigmoid(self.output)
                sf=sigmoid.forward()
                return sf
    
    def MaxPooling2D(self,inputs,kernel=(1,1)):
        max_pool=si.block_reduce(inputs,kernel,np.max)
        return max_pool
        
            
    def Flatten(self,inputs,flatten_type='C'):
            return inputs.flatten(flatten_type)
    
    def BackPropogation(self):
        for i in range(len(self.weights_list)):
            for j in range(len(self.weights_list[0])):
                   pass 
                
    def Predict(self):
        self.test_input=self.inputs
        arr=np.array(output_list[0][0])
        #arr=arr.flatten('C')
        #print(arr)
        #print(arr.shape)
        #test_input=test_input.Flatten('C')
        output=0
        output_label=[]
        for k in range(0,self.test_input.shape[0]):
            for i in range(0,arr.shape[0]):
                output=output+self.test_input[k][i]+arr[i]
                if(output.any()<0):
                    output=0
                    output_label.append(output)
                else:
                    output_label.append(output)
                    
        return output_label
        

            

    
