import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:image/image.dart' as imglib;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Cat and Dog Classifier',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSwatch(primarySwatch: Colors.deepPurple),
      ),
      home: const MyHomePage(title: 'Cat and Dog Classifier'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  File? _image;
  final picker = ImagePicker();
  String _result = "Awaiting classification...";
  late tfl.Interpreter _interpreter;
  late int _inputSize;

  @override
  void initState() {
    super.initState();
    loadModel();
  }

  Future<void> loadModel() async {
    try {
      _interpreter = await tfl.Interpreter.fromAsset('assets/cat_dog_model.tflite');
      var inputShape = _interpreter.getInputTensor(0).shape;
      _inputSize = inputShape[1]; // Assuming input shape is [1, height, width, channels]
    } catch (e) {
      print('Failed to load model: $e');
    }
  }

  Future<void> selectImage() async {
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);

    setState(() {
      if (pickedFile != null) {
        _image = File(pickedFile.path);
        classifyImage(_image!);
      } else {
        print('No image selected.');
      }
    });
  }

  Future<List<List<List<List<int>>>>> imageToMatrix(File imageFile) async {
    // Read image bytes
    Uint8List bytes = await imageFile.readAsBytes();

    // Decode image using image package
    imglib.Image? image = imglib.decodeImage(bytes);

    if (image != null) {
      // Resize image to 150x150
      imglib.Image resizedImage = imglib.copyResize(image, width: 150, height: 150);

      // Convert resized image to 4D matrix
      List<List<List<List<int>>>> imageMatrix = [];

      // Create batch dimension (batch_size = 1)
      List<List<List<int>>> batch = [];

      for (int y = 0; y < resizedImage.height; y++) {
        List<List<int>> row = [];
        for (int x = 0; x < resizedImage.width; x++) {
          int pixel = resizedImage.getPixel(x, y);
          int r = imglib.getRed(pixel);
          int g = imglib.getGreen(pixel);
          int b = imglib.getBlue(pixel);
          row.add([r, g, b]);
        }
        batch.add(row);
      }

      imageMatrix.add(batch);

      return imageMatrix;
    } else {
      throw Exception('Failed to decode image.');
    }
  }

  Future<void> classifyImage(File imageFile) async {
    print('Classification...');
    if (_interpreter == null) {
      print('Interpreter not initialized.');
      return;
    }

    try {
      // Preprocess the image
      var input = await imageToMatrix(imageFile!);
      var output = List.generate(1, (_) => List.filled(1, 0.0));

      // Run inference
      print('>>>input shape: ${_interpreter!. getInputTensor(0).shape},type: ${_interpreter!. getInputTensor(0).type}');
      _interpreter.run(input, output);
      double dogProbability = output[0][0];
      double catProbability = 1 - output[0][0];
      String predictedClass = dogProbability > catProbability
          ? 'I am ${dogProbability.toStringAsFixed(2)}% certain this is a dog.'
          : 'I am ${catProbability.toStringAsFixed(2)}% certain this is a cat.';

      setState(() {
        _result = predictedClass;
      });
    } catch (e) {
      print('Failed to classify image: $e');
      setState(() {
        _result = 'Failed to classify image: $e';
      });
    }
  }

  @override
  void dispose() {
    super.dispose();
    _interpreter.close();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            _image == null
                ? const Text('No image selected.')
                : Image.file(_image!),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: selectImage,
              child: const Text('Select Image'),
            ),
            const SizedBox(height: 20),
            Text(
              _result,
              style: Theme.of(context).textTheme.headline6,
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}