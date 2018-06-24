from datetime import datetime

date_object = datetime.strptime("2013-02-01 08:00","%Y-%m-%d %H:%M")
test = datetime(2013, 1, 1)
delta = date_object - test