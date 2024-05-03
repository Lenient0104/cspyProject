class User:
    def __init__(self, weight, driving_license, pal, age, preference=None):
        if preference is None:
            preference = ['e_bike_1', 'e_scooter_1', 'e_car']
        self.weight = weight
        self.driving_license = driving_license
        self.pal = pal
        self.age = age
        self.preference = preference

        # self.start = start
        # self.destination = destination



