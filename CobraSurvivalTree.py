from sksurv.tree import SurvivalTree

class CobraSurvivalTree(SurvivalTree):
    """
    Version of Survival Tree for PyCobra where predict method calculates
    the mean survival time of X instead of the risk score
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def predict(self, X):
        surv_func = super().predict_survival_function(X,return_array=True)
        ans = [0]*len(surv_func)
        for i in range(len(surv_func)):
            ans[i] = sum((self.event_times_[:-1]+self.event_times_[1:])*(surv_func[i,:-1]-surv_func[i,1:]))/2
            ans[i] += self.event_times_[-1]*surv_func[i,-1]
        return ans
    
    def predict_surv(self, X, return_array=True):
        return super().predict_survival_function(X,return_array=return_array)
    
    def fit(self, *args, **kwargs):
        return super().fit(*args, **kwargs)