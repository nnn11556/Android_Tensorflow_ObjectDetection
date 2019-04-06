package com.example.tf_objectdetection;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.objectdetectorlib.Classifier;
import com.example.objectdetectorlib.TensorflowObjectDetector;

import java.util.List;

public class MainActivity extends AppCompatActivity
{

    private ImageView imgView;
    private TextView textView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imgView = (ImageView) findViewById(R.id.imageView);
        textView = (TextView) findViewById(R.id.textView);

        Bitmap imageBitMap = TensorflowObjectDetector.getBitmapFromAssetsFolder(this, "person.jpg");
        imgView.setImageBitmap(imageBitMap);
        Bitmap tmpImageBitMap = imageBitMap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(tmpImageBitMap);
        Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setTextSize(60);
        paint.setTextAlign(Paint.Align.LEFT);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(5);

        List<Classifier.Recognition> mappedRecognitions = TensorflowObjectDetector.processing(this, imageBitMap);
        for (Classifier.Recognition result : mappedRecognitions) {
            TensorflowObjectDetector.drawObjectResult(result, canvas, paint);
            textView.append(result.toString()+"\n");
        }
        imgView.setImageBitmap(tmpImageBitMap);
    }
}
