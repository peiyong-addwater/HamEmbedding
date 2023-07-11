import Augmentor

p = Augmentor.Pipeline("images")

p.rotate(probability=0.9, max_left_rotation=10, max_right_rotation=10)
p.random_distortion(probability=0.9, grid_width=4, grid_height=4, magnitude=8)
p.sample(100)