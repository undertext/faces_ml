import tensorflow as tf
import tensorflow_addons as tfa


class FacesModel:

    @staticmethod
    def get_compiled_model(image_width: int, image_height: int, faces_count: int) -> tf.keras.Model:

        def fire_layer(model: tf.keras.Model, squeeze_number: int, output_number: int):
            model.add(tf.keras.layers.Conv2D(squeeze_number, 1, padding='same', activation='relu'))
            model.add(tf.keras.layers.Conv2D(output_number, 3, padding='same', activation='relu'))

        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu',
                                   input_shape=(image_width, image_height, 3)),
            tf.keras.layers.MaxPooling2D(2),
        ])

        model.add(tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(2))

        fire_layer(model, 16, 64)
        model.add(tf.keras.layers.MaxPooling2D(2))
        fire_layer(model, 32, 128)
        model.add(tf.keras.layers.MaxPooling2D(2))
        fire_layer(model, 64, 256)
        model.add(tf.keras.layers.MaxPooling2D(2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(activation="relu", units=128))
        model.add(tf.keras.layers.Dense(activation="relu", units=64))
        model.add(tf.keras.layers.Dense(activation="sigmoid", units=faces_count * 4))

        @tf.function
        def bb_intersection_over_union(boxesA, boxesB):
            # determine the (x, y)-coordinates of the intersection rectangle
            # xA = tf.math.reduce_max([boxesA, boxesB], axis=0)
            # print(xA.shape)
            i = 0
            iou = tf.constant(0.0)
            ha = tf.constant(0.0)
            for boxA in boxesA:
                boxB = boxesB[i]
                xA = tf.math.maximum(boxA[0], boxB[0])
                yA = tf.math.maximum(boxA[1], boxB[1])
                xB = tf.math.minimum(boxA[0] + boxA[2], boxB[0] + boxB[2])
                yB = tf.math.minimum(boxA[1] + boxA[3], boxB[1] + boxB[3])
                # compute the area of intersection rectangle
                interArea = tf.math.maximum(0.0, xB - xA) * tf.math.maximum(0.0, yB - yA)
                ha += interArea
                # compute the area of both the prediction and ground-truth
                # rectangles
                boxAArea = (boxA[2]) * (boxA[3])
                boxBArea = (boxB[2]) * (boxB[3])
                # compute the intersection over union by taking the intersection
                # area and dividing it by the sum of prediction + ground-truth
                # areas - the interesection area
                sec = interArea / (boxAArea + boxBArea - interArea)

                if tf.math.is_finite(sec):
                    iou += sec
                # print(iou)
                i += 1
                # return the intersection over union value
            return iou / float(i)

        def euclideanDistance(x, y):
            dist = tf.sqrt(tf.reduce_sum(tf.square(x - y)))
            return dist

        def metric_fn(a, b):
            return bb_intersection_over_union(tf.reshape(a, [-1, 4]), tf.reshape(b, [-1, 4]))

        def loss_fn2(a, b):
            def _fn(boxA, boxB):
                gl = tfa.losses.GIoULoss(mode="iou")
                centerA = tf.stack([boxA[0] + boxA[2] / 2, boxA[1] + boxA[3] / 2], axis=0)
                centerB = tf.stack([boxB[0] + boxB[2] / 2, boxB[1] + boxB[3] / 2], axis=0)
                return euclideanDistance(centerA, centerB) + gl(
                    [[boxA[1], boxA[0], boxA[1] + boxA[3], boxA[0] + boxA[2]]],
                    [[boxB[1], boxB[0], boxB[1] + boxB[3], boxB[0] + boxB[2]]])

            reshaped = tf.concat([tf.reshape(a, [-1, 4]), tf.reshape(b, [-1, 4])], axis=1)
            return tf.reduce_sum(tf.map_fn(lambda ab: _fn(ab[0:4], ab[4:8]), reshaped)) / faces_count

        model.compile(optimizer='adam',
                      run_eagerly=True,
                      loss=loss_fn2,
                      metrics=[metric_fn])
        return model
