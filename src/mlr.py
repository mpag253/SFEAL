
class EiMLR:
    def __init__(self, age, fvc, dlco, bmi, rvtlc, tlc):
        
        self.m1 = {
            'b0': 1.3773,
            'b1': -0.0364,
            'b2': 0.3754,
            'b3': -0.0409
        }

        self.m2 = {
            'b0': 3.4902,
            'b1': -0.1642,
            'b2': 0.0167,
        }

        self.m3 = {
            'b0': 4.8953,
            'b1': -0.018,
            'b2': -0.4557,
            'b3': -0.0506
        }

        self.age = age
        self.fvc = fvc
        self.dlco = dlco
        self.bmi = bmi
        self.rvtlc = rvtlc
        self.tlc = tlc

    def predict_m1(self):
        return self.m1['b0'] + \
               (self.m1['b1'] * self.age) + \
               (self.m1['b2'] * self.fvc) + \
               (self.m1['b3'] * self.dlco)

    def predict_m2(self):
        return self.m2['b0'] + \
               (self.m2['b1'] * self.bmi) + \
               (self.m2['b2'] * self.rvtlc)

    def predict_m3(self):
        return self.m3['b0'] + \
               (self.m3['b1'] * self.age) + \
               (self.m3['b2'] * self.tlc) + \
               (self.m3['b3'] * self.dlco)


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
