import tensorflow as tf


# How to changeing variable value and making simple counter :

state = tf.Variable(0)
one = tf.constant(1)
new_value = tf.add(one , state)
update = update = tf.assign(state, new_value)
init_op= tf.global_variables_initializer()

with tf.Session() as session:
  session.run(init_op)
  print(session.run(state))
  for _ in range(3):
    session.run(update)
    print(session.run(state))




