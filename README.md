# Animal_Gender


# ABC-Animal-Behaviour
Predict gender of birds from their trajectories.

The code here provides all the the things which I tried.

# Feature Engineering(what worked and didn't worked)

- Harvesine distance between maximum {latitude,longitude} and minimum {latitude,longitude} and multiplied it by 2( Assuming that distance will be 2 times for arrival and departure from a reference point)( Tried computing harvesine distance between each of the consequtive points and summing them up but it didn't worked due to too much noise)

- Minimum/Maximum/Mean {latitude,longitude} . (Tried variance too but it didn't worked in improving the results).

- Tried computing area of polygon of trajectory but it didn't worked so wan't included.

- Tried counting dives and elevations as a feature but didn't worked.

- Feature that worked most was length of stay(which means no of times bird stayed in journey)

- Also at the end of competition features like left to right extent (difference) gave some good improvement.

- One important thing to note here was that some of features were normalized with maximum no. of days travelled by the bird which was major part of the improvement.




# Library Requirements:
Do run if not installed :   pip install library name
- numpy
- pandas
- mlbox
- pyproj
- shapely
- functools
- scipy
- csv

# Usage:
- Put all the data(train and test folders) inside data folder.
- To run python automl.py

