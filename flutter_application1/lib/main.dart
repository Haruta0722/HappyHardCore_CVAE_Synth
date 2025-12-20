import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: BlePage(),
    );
  }
}

class BlePage extends StatefulWidget {
  const BlePage({super.key});

  @override
  State<BlePage> createState() => _BlePageState();
}

class _BlePageState extends State<BlePage> {
  final List<ScanResult> _scanResults = [];
  BluetoothDevice? _connectedDevice;

  // Jetson 側と合わせる UUID
  final Guid serviceUuid =
      Guid("12345678-1234-5678-1234-56789abcdef0");
  final Guid charUuid =
      Guid("12345678-1234-5678-1234-56789abcdef1");

  @override
  void initState() {
    super.initState();
    _startScan();
  }

  void _startScan() {
    _scanResults.clear();

    FlutterBluePlus.startScan(timeout: const Duration(seconds: 5));

    FlutterBluePlus.scanResults.listen((results) {
      for (final r in results) {
        if (_scanResults.every((e) => e.device.id != r.device.id)) {
          setState(() {
            _scanResults.add(r);
          });
        }
      }
    });
  }

  Future<void> _connect(BluetoothDevice device) async {
    await device.connect(autoConnect: false);
    setState(() {
      _connectedDevice = device;
    });
  }

  Future<void> _sendText(String text) async {
    if (_connectedDevice == null) return;

    final services = await _connectedDevice!.discoverServices();
    final service =
        services.firstWhere((s) => s.uuid == serviceUuid);
    final characteristic =
        service.characteristics.firstWhere((c) => c.uuid == charUuid);

    await characteristic.write(utf8.encode(text), withoutResponse: true);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("BLE Demo")),
      body: Column(
        children: [
          const Padding(
            padding: EdgeInsets.all(8),
            child: Text("Scan Results"),
          ),
          Expanded(
            child: ListView.builder(
              itemCount: _scanResults.length,
              itemBuilder: (context, index) {
                final r = _scanResults[index];
                final name =
                    r.device.name.isNotEmpty ? r.device.name : "Unknown";

                return ListTile(
                  title: Text(name),
                  subtitle: Text(r.device.id.toString()),
                  onTap: () => _connect(r.device),
                );
              },
            ),
          ),
          const Divider(),
          ElevatedButton(
            onPressed: _connectedDevice == null
                ? null
                : () => _sendText("Hello Jetson"),
            child: const Text("Send Text"),
          ),
          const SizedBox(height: 16),
        ],
      ),
    );
  }
}

