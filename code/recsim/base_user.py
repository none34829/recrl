"""
Simple stochastic user model for CTR simulation without RecSim-NG dependencies.

CTR ≈ sigmoid(alpha * ranker_score + beta * expl_quality + user_bias)
"""

import tensorflow as tf

class SimpleUser:
    def __init__(self, sigma: float = 0.3):
        self.sigma = sigma

    def initial_state(self):
        return {"user_bias": tf.random.normal([], stddev=self.sigma, dtype=tf.float32)}

    def next_state(self, previous_state, slate, responses, time_step):
        # Memoryless user for simplicity
        return previous_state

    def next_response(self, state, slate_docs, scores, expl_quality):
        alpha, beta = 1.0, 0.5
        logits = alpha * scores + beta * expl_quality + state["user_bias"]
        probs = tf.squeeze(tf.math.sigmoid(logits))
        rand = tf.random.uniform(tf.shape(probs), dtype=probs.dtype)
        click = tf.cast(rand < probs, tf.float32)
        return {"clicked": click}
