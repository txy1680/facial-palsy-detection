package com.example.opencvdemo;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.ContentResolver;
import android.content.Context;
import android.content.Intent;
import android.content.res.AssetManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLConnection;

import static android.content.pm.PackageManager.PERMISSION_GRANTED;


public class MainActivity extends AppCompatActivity implements View.OnClickListener {
    //定义两个控件，分别用来显示图像和文本
    private ImageView imageView;
    private TextView textView;
    Uri uri;//统一资源标识符
    String url;//统一资源定位符
    Bitmap bitmap=null;
    // 加载生成的动态链接库
    static {
        System.loadLibrary("native-lib");
    }
    // 声明JNI函数，对应native-lib.cpp里定义的函数
    native String mnistJNI(Object bitmap, String str);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageView = findViewById(R.id.imageView);
        findViewById(R.id.show).setOnClickListener(this);
        findViewById(R.id.process).setOnClickListener(this);
        findViewById(R.id.choose_images).setOnClickListener(this);

        textView = findViewById(R.id.textView);
        findViewById(R.id.textView).setOnClickListener(this);

    }

    @Override
    public void onClick(View v) {
        //如果点击了选择图片，就执行
        if (v.getId() == R.id.choose_images)
        {

            Intent intent = new Intent(Intent.ACTION_PICK, null);
            intent.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*");
            startActivityForResult(intent, 2);//
        }
        //如果点击了process按钮，就进入处理
        if (v.getId() == R.id.process){
            Bitmap bitmap= null;//定义一个空的bitmap
            try {
                bitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(uri));
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
            //把模型从Assets拷贝到Cache，再获取模型的路径，供C++读取
            copyAssetAndWrite("best.mnn");
            File dataFile = new File(getCacheDir(),"best.mnn");
            //System.out.println("模型路径:" + dataFile.getAbsolutePath());
            Bitmap new_bitamp = getNewBitmap(bitmap, 640, 640);
            imageView.setImageBitmap(new_bitamp);
            textView.setText(mnistJNI(new_bitamp, dataFile.getAbsolutePath()));
        }
    }
    public void onActivityResult(int requestCode, int resultCode, Intent data) {

        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 2) {
            // 从相册返回的数据
            if (data != null) {
                // 得到图片的全路径
                uri = data.getData();

                Context mContext = this;
                url = getRealFilePath(mContext, uri);


            }
        }
    }

    public Bitmap getNewBitmap(Bitmap bitmap, int newWidth ,int newHeight){
        // 获得图片的宽高.
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        // 计算缩放比例.
        float scaleWidth = ((float) newWidth) / width;
        float scaleHeight = ((float) newHeight) / height;
        // 取得想要缩放的matrix参数.
        Matrix matrix = new Matrix();
        matrix.postScale(scaleWidth, scaleHeight);
        // 得到新的图片.
        Bitmap newBitmap = Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, true);
        return newBitmap;
    }

    public static Bitmap getBitMBitmap(String urlpath) {
        Bitmap map = null;
        try {
            FileInputStream fileInputStream = new FileInputStream(urlpath);
            if(fileInputStream != null) {
                BitmapFactory.Options options = new BitmapFactory.Options();
                options.inSampleSize = 1; //当图片资源太大的适合，会出现内存溢出。图片宽高都为原来的二分之一，即图片为原来的四分一
                map = BitmapFactory.decodeStream(fileInputStream, null, options);
            }
        } catch(Exception e) {
            e.printStackTrace();
        }
        return map;
    }
    public static String getRealFilePath(final Context context, final Uri uri ) {
        if ( null == uri ) return null;
        final String scheme = uri.getScheme();
        String data = null;
        if ( scheme == null )
            data = uri.getPath();
        else if ( ContentResolver.SCHEME_FILE.equals( scheme ) ) {
            data = uri.getPath();
        } else if ( ContentResolver.SCHEME_CONTENT.equals( scheme ) ) {
            Cursor cursor = context.getContentResolver().query( uri, new String[] { MediaStore.Images.ImageColumns.DATA }, null, null, null );
            if ( null != cursor ) {
                if ( cursor.moveToFirst() ) {
                    int index = cursor.getColumnIndex( MediaStore.Images.ImageColumns.DATA );
                    if ( index > -1 ) {
                        data = cursor.getString( index );
                    }
                }
                cursor.close();
            }
        }
        return data;
    }
    @Override
    public void onPointerCaptureChanged(boolean hasCapture) {
    }
    private boolean copyAssetAndWrite(String fileName){
        try {
            File cacheDir=getCacheDir();
            if (!cacheDir.exists()){
                cacheDir.mkdirs();
            }
            File outFile =new File(cacheDir,fileName);
            if (!outFile.exists()){
                boolean res=outFile.createNewFile();
                if (!res){
                    return false;
                }
            }else {
                if (outFile.length()>10){//表示已经写入一次
                    return true;
                }
            }
            InputStream is=getAssets().open(fileName);
            FileOutputStream fos = new FileOutputStream(outFile);
            byte[] buffer = new byte[1024];
            int byteCount;
            while ((byteCount = is.read(buffer)) != -1) {
                fos.write(buffer, 0, byteCount);
            }
            fos.flush();
            is.close();
            fos.close();
            return true;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return false;
    }
}
