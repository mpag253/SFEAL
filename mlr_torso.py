
class EiMLR:
    def __init__(self, age, sex, height, weight, bmi, fvc, fev1, fev1fvc, frc, tlc, rv, fef, pefr, rvtlc, dlco):
        
        self.m1 = {
            'b0':  5.2107,
            'b1': -0.04600,
            'b2': -0.5998,
            'b3': -0.05721,
            'b4': -0.03592,
            'bse0': 1.0173,
            'bse1': 0.006048,
            'bse2': 0.2127,
            'bse3': 0.02655,
            'bse4': 0.01734,
        }

        self.m2 = {
            'b0':  0.,
            'bse0': 0.,
        }

        self.m3 = {
            'b0': 13.906,
            'b1': -1.1143,
            'b2': -9.7762,
            'b3':  0.06008,
            'b4': -0.03506,
            'bse0': 3.110,
            'bse1': 0.3009,
            'bse2': 1.9854,
            'bse3': 0.01300,
            'bse4': 0.01348,
        }
        
        self.m4 = {
            'b0': 13.617,
            'b1': -0.1880,
            'b2': -2.5641,
            'b3':  4.0449,
            'b4': -0.1121,
            'b5': -0.3669,
            'b6': -0.6513,
            'bse0': 4.0885,
            'bse1': 0.03876,
            'bse2': 0.9805,
            'bse3': 1.4029,
            'bse4': 0.05150,
            'bse5': 0.1688,
            'bse6': 0.2232,
        } 
        
        self.m5 = {
            'b0':  4.9116,
            'b1': -0.01458,
            'b2': -0.5095,
            'b3': -0.4046,
            'b4': -0.2768,
            'b5': -0.08238,
            'bse0': 0.4217,
            'bse1': 0.004679,
            'bse2': 0.1297,
            'bse3': 0.1132,
            'bse4': 0.1205,
            'bse5': 0.04096,
        }

        self.age = age
        self.sex = sex
        self.height = height
        self.weight = weight
        self.bmi = bmi
        self.fvc = fvc
        self.fev1 = fev1
        self.fev1fvc = fev1fvc
        self.frc = frc
        self.tlc = tlc
        self.rv = rv
        self.fef = fef
        self.pefr = pefr
        self.rvtlc = rvtlc
        self.dlco = dlco


    def predict_m1(self):
        return self.m1['b0'] + \
               (self.m1['b1'] * self.age) + \
               (self.m1['b2'] * self.sex) + \
               (self.m1['b3'] * self.bmi) + \
               (self.m1['b4'] * self.dlco)

    def predict_m2(self):
        return self.m2['b0']

    def predict_m3(self):
        return self.m3['b0'] + \
               (self.m3['b1'] * self.sex) + \
               (self.m3['b2'] * self.height) + \
               (self.m3['b3'] * self.weight) + \
               (self.m3['b4'] * self.rvtlc)
               
    def predict_m4(self):
        return self.m4['b0'] + \
               (self.m4['b1'] * self.bmi) + \
               (self.m4['b2'] * self.fvc) + \
               (self.m4['b3'] * self.fev1) + \
               (self.m4['b4'] * self.fev1fvc) + \
               (self.m4['b5'] * self.rv) + \
               (self.m4['b6'] * self.fef)
               
    def predict_m5(self):
        return self.m5['b0'] + \
               (self.m5['b1'] * self.age) + \
               (self.m5['b2'] * self.fev1) + \
               (self.m5['b3'] * self.frc) + \
               (self.m5['b4'] * self.rv) + \
               (self.m5['b5'] * self.pefr)
                  

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
