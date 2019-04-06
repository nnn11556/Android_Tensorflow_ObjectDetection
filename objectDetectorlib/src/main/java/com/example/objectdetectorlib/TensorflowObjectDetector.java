package com.example.objectdetectorlib;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.os.SystemClock;
import android.widget.Toast;

import com.example.objectdetectorlib.env.ImageUtils;
import com.example.objectdetectorlib.env.Logger;

import java.io.IOException;
import java.io.InputStream;
import java.util.LinkedList;
import java.util.List;

public class TensorflowObjectDetector {

    private static int rotation = 0;

    private static int previewWidth = 0;
    private static int previewHeight = 0;

    private static final Logger LOGGER = new Logger();

    // Configuration values for object detection model.
    //image input size
    private static final int TF_OD_API_INPUT_SIZE = 1000;
    /*
     per image 300*300
            model                 inference time         accuracy
        ssd-mobilenet                  800ms               low
     faster-rcnn-inception             4500ms              high
      */
    // model file path(support ssd and faster-rcnn
    private static final String TF_OD_API_MODEL_FILE = "file:///android_asset/frozen_inference_graph.pb";;
    // detection label path
    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/coco_labels_list.txt";
    //confindence value,generally need above 0.5f
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.8f;
    // is need keep aaspect?
    private static final boolean MAINTAIN_ASPECT = false;
    // not use
    private static Integer sensorOrientation;
    //detctor
    public static Classifier detector;
    //processing time per image
    private static long lastProcessingTimeMs;
    private static Bitmap rgbFrameBitmap = null;
    private static Bitmap croppedBitmap = null;

    private static long timestamp = 0;

    private static Matrix frameToCropTransform;
    private static Matrix cropToFrameTransform;

    /*
     *  object detection main function
     */
    public static List<Classifier.Recognition> processing(Context context, Bitmap imageBitmap){
        int imageWidth = imageBitmap.getWidth();
        int imageHeight = imageBitmap.getHeight();
        LOGGER.i("image size:"+imageWidth+","+imageHeight);
        if(initialiseDetector(context, imageWidth, imageHeight, rotation)){
            return processImage(imageBitmap);
        }
        return null;
    }

    /*
     *  draw the detection result on image
     */
    public static void drawObjectResult(Classifier.Recognition result, Canvas canvas, Paint paint){
        //draw the object edge rect
        RectF rectF = result.getLocation();
        canvas.drawRect(rectF, paint);
        //draw the object label(class and confidence)
        String label = "";
        label += result.getTitle() + " ";
        label += String.format("(%.1f%%) ", result.getConfidence() * 100.0f);
        canvas.drawText(label, (int)(rectF.left), (int)(rectF.top), paint);
    }

    /*
     *   read the file from asset folder as bitmap format
     */
    public static Bitmap getBitmapFromAssetsFolder(Context context, String fileName) {
        Bitmap bitmap = null;
        try
        {
            InputStream istr=context.getAssets().open(fileName);
            bitmap= BitmapFactory.decodeStream(istr);
        }
        catch (IOException e)
        {
            System.out.println("Error: " + e);
            System.exit(0);
        }
        return bitmap;
    }

    private static int getScreenOrientation() { return 0; }

    /*
     *   initial the object detector
     */
    private static boolean initialiseDetector(Context context, final int imageWidth, final int imageHeight, final int rotation) {

        int cropSize;
        try
        {
            detector = TensorFlowObjectDetectionAPIModel.create(context.getAssets(), TF_OD_API_MODEL_FILE, TF_OD_API_LABELS_FILE, TF_OD_API_INPUT_SIZE);
            cropSize = TF_OD_API_INPUT_SIZE;
        }
        catch (final IOException e)
        {
            LOGGER.e("Exception initializing classifier!", e);
            Toast toast = Toast.makeText(context.getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            return false;
        }

        previewWidth = imageWidth;
        previewHeight = imageHeight;

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888);
        frameToCropTransform = ImageUtils.getTransformationMatrix(previewWidth, previewHeight, cropSize, cropSize, sensorOrientation, MAINTAIN_ASPECT);
        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        return true;
    }

    /*
        process one image by object detection model
     */
    private static List<Classifier.Recognition> processImage(Bitmap currentFrameBmpTemp){

        ++timestamp;
        final long currTimestamp = timestamp;

        LOGGER.i("processImage");
        LOGGER.i("Preparing image " + currTimestamp);

        rgbFrameBitmap = currentFrameBmpTemp;
        Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
//        // For examining the actual TF input.
        LOGGER.i("Running detection on image " + currTimestamp);
        final long startTime = SystemClock.uptimeMillis();
        final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
        LOGGER.i("Process image during "+lastProcessingTimeMs+"ms");
        float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;

        final List<Classifier.Recognition> mappedRecognitions = new LinkedList<Classifier.Recognition>();

        for (final Classifier.Recognition result : results)
        {
            final RectF location = result.getLocation();
            LOGGER.i("object found");
            if (location != null && result.getConfidence() >= minimumConfidence)
            {
                LOGGER.i("passedConfidence");
                LOGGER.i("result: "+result.toString());
                cropToFrameTransform.mapRect(location);
                result.setLocation(location);
                mappedRecognitions.add(result);
            }
        }
        return mappedRecognitions;
    }

}
