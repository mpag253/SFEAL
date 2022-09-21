
class EiMLR:
    #def __init__(self, age, sex, height, weight, bmi, fvc, fev1, fev1fvc, frc, tlc, rv, fef, pefr, rvtlc, dlco):
    def __init__(self, mlr_dict, subject_data_cols):
         
        # Subject information
        self.age = subject_data_cols['Age']
        self.sex = subject_data_cols['Sex']
        self.height = subject_data_cols['Height']
        self.weight = subject_data_cols['Weight']
        self.fvc = subject_data_cols['FVC']
        self.fev1 = subject_data_cols['FEV1']
        self.tlc = subject_data_cols['TLC']
        self.rv = subject_data_cols['RV']
        self.fef = subject_data_cols['FEF']
        
        # MLR coefficients
        self.m1 = mlr_dict['parameters']['M1']
        self.m2 = mlr_dict['parameters']['M2']
        self.m3 = mlr_dict['parameters']['M3']
        self.m4 = mlr_dict['parameters']['M4']
        self.m5 = mlr_dict['parameters']['M5']

    def predict_m1(self):
        return self.m1['Intercept'] + (self.m1['Age'] * self.age) + \
                                      (self.m1['Sex'] * self.sex) + \
                                      (self.m1['Height'] * self.height) + \
                                      (self.m1['Weight'] * self.weight) + \
                                      (self.m1['FVC'] * self.fvc) + \
                                      (self.m1['FEV1'] * self.fev1) + \
                                      (self.m1['TLC'] * self.tlc) + \
                                      (self.m1['RV'] * self.rv) + \
                                      (self.m1['FEF'] * self.fef)

    def predict_m2(self):
        return self.m2['Intercept'] + (self.m2['Age'] * self.age) + \
                                      (self.m2['Sex'] * self.sex) + \
                                      (self.m2['Height'] * self.height) + \
                                      (self.m2['Weight'] * self.weight) + \
                                      (self.m2['FVC'] * self.fvc) + \
                                      (self.m2['FEV1'] * self.fev1) + \
                                      (self.m2['TLC'] * self.tlc) + \
                                      (self.m2['RV'] * self.rv) + \
                                      (self.m2['FEF'] * self.fef)
                                      
    def predict_m3(self):
        return self.m3['Intercept'] + (self.m3['Age'] * self.age) + \
                                      (self.m3['Sex'] * self.sex) + \
                                      (self.m3['Height'] * self.height) + \
                                      (self.m3['Weight'] * self.weight) + \
                                      (self.m3['FVC'] * self.fvc) + \
                                      (self.m3['FEV1'] * self.fev1) + \
                                      (self.m3['TLC'] * self.tlc) + \
                                      (self.m3['RV'] * self.rv) + \
                                      (self.m3['FEF'] * self.fef)
                                      
    def predict_m4(self):
        return self.m4['Intercept'] + (self.m4['Age'] * self.age) + \
                                      (self.m4['Sex'] * self.sex) + \
                                      (self.m4['Height'] * self.height) + \
                                      (self.m4['Weight'] * self.weight) + \
                                      (self.m4['FVC'] * self.fvc) + \
                                      (self.m4['FEV1'] * self.fev1) + \
                                      (self.m4['TLC'] * self.tlc) + \
                                      (self.m4['RV'] * self.rv) + \
                                      (self.m4['FEF'] * self.fef)
                                      
    def predict_m5(self):
        return self.m5['Intercept'] + (self.m5['Age'] * self.age) + \
                                      (self.m5['Sex'] * self.sex) + \
                                      (self.m5['Height'] * self.height) + \
                                      (self.m5['Weight'] * self.weight) + \
                                      (self.m5['FVC'] * self.fvc) + \
                                      (self.m5['FEV1'] * self.fev1) + \
                                      (self.m5['TLC'] * self.tlc) + \
                                      (self.m5['RV'] * self.rv) + \
                                      (self.m5['FEF'] * self.fef)
               
    def predict_modes(self):
        ei_m1 = self.predict_m1()
        ei_m2 = self.predict_m2()
        ei_m3 = self.predict_m3()
        ei_m4 = self.predict_m4()
        ei_m5 = self.predict_m5()
        return [ei_m1, ei_m2, ei_m3, ei_m4, ei_m5]
                  

class EeMLR:
    def __init__(self, age, frc, rv, rvtlc, dlco, fev1, pefr, vc):

        self.m1 = {
            'b0': 0.7802,
            'b1': -0.0257,
            'b2': 0.4243,
            'b3': -0.359
        }

        self.m2 = {
            'b0': 4.1438,
            'b1': -0.0383,
            'b2': -0.093,
            'b3': -0.5009,
            'b4': 0.572,
            'b5': -0.1269,
        }

        self.m3 = {
            'b0': 3.3263,
            'b1': -0.0317,
            'b2': -0.5316,
            'b3': 0.2464
        }

        self.age = age
        self.frc = frc
        self.rv = rv
        self.rvtlc = rvtlc
        self.dlco = dlco
        self.fev1 = fev1
        self.pefr = pefr
        self.vc = vc

    def predict_m1(self):
        return self.m1['b0'] + \
               (self.m1['b1'] * self.age) + \
               (self.m1['b2'] * self.frc) + \
               (self.m1['b3'] * self.rv)

    def predict_m2(self):
        return self.m2['b0'] + \
               (self.m2['b1'] * self.rvtlc) + \
               (self.m2['b2'] * self.dlco) + \
               (self.m2['b3'] * self.frc) + \
               (self.m2['b4'] * self.fev1) + \
               (self.m2['b5'] * self.pefr)

    def predict_m3(self):
        return self.m3['b0'] + \
               (self.m3['b1'] * self.age) + \
               (self.m3['b2'] * self.vc) + \
               (self.m3['b3'] * self.rv)
