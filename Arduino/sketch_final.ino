#include <Stepper.h>
#define CO2SENSOR_DEBUG false
#define CO2_LOW 600
#define CO2_HIGHT 1000


const int co2_b = 600;
const int co2_d = 400;

class CO2Sensor
{
  public:
    CO2Sensor(int analogPin);
    CO2Sensor(int analogPin, float inertia, int tries);
    int read();
    void calibrate();

    int getVoltage();

    int getGreenLevel();
    int getRedLevel();

  private:
    void init();

    int _analogPin;
    int _inertia;
    int _tries;
    int _co2_v;
    int _greenLevel;
    double _co2_a;
    double _co2ppm;
};

CO2Sensor::CO2Sensor(int analogPin){
  _inertia = 0.99;
  _tries = 3;
  _analogPin = analogPin;
  init();
}

CO2Sensor::CO2Sensor(int analogPin, float inertia, int tries){
  _inertia = inertia;
  _tries = tries;
  _analogPin = analogPin;
  init();
}

int CO2Sensor::read(){
  int v = 0;

  analogRead(_analogPin);
  for (int i = 0; i < _tries; i++)
  {
     v += analogRead(_analogPin);
     delay(20);
  }
  _co2_v = (1-_inertia)*(v*5000.0)/(1024.0*_tries)+_co2_v*_inertia;

  double co2_exp = (_co2_a-_co2_v)/co2_b;

  _co2ppm = pow(co2_d, co2_exp);

  #if CO2SENSOR_DEBUG
  Serial.print("Exponent: ");
  Serial.println(co2_exp);

  Serial.println("CO2 == ");

  Serial.print(_co2_v);
  Serial.println(" mV");
  Serial.print(_co2ppm);
  Serial.println(" ppm");
  #endif

  if (_co2ppm<CO2_LOW) _greenLevel = 255;
  else {
    if (_co2ppm>CO2_HIGHT) _greenLevel = 0;
    else _greenLevel = map(_co2ppm, CO2_LOW, CO2_HIGHT, 255, 0);
  }

  return _co2ppm;
}

void CO2Sensor::calibrate(){
  read();

  #if CO2SENSOR_DEBUG
  Serial.print("Calibration. Old a: ");
  Serial.print(_co2_a);
  #endif

  _co2_a = _co2_v + co2_b;
  _co2ppm = co2_d;

  #if CO2SENSOR_DEBUG
  Serial.print(", New a: ");
  Serial.println(_co2_a);
  #endif
}

void CO2Sensor::init(){
  _co2_a = 1500;
  _co2ppm = co2_d;
}

int CO2Sensor::getVoltage(){
  return _co2_v;
}

int CO2Sensor::getGreenLevel(){
  return _greenLevel;
}

int CO2Sensor::getRedLevel(){
  return 255-_greenLevel;
}
//#include "CO2Sensor.h"
#define MG_PIN (0)

CO2Sensor co2Sensor(A0, 0.99, 100);
int led=7;
int val =0;
const int stepsPerRevolution = 2048;
Stepper myStepper(stepsPerRevolution,11,9,10,8);

void setup() {
Serial.begin(9600);
co2Sensor.calibrate();
myStepper.setSpeed(14); 
pinMode(led, OUTPUT);
}

void loop() {
  int hodnota = co2Sensor.read();
  if(Serial.available()){
    val=Serial.parseInt();
    analogWrite(led,255);
    delay(2000);
  }
  else{
    analogWrite(led,0);
  }
 Serial.print("CO2 : ");
 Serial.print(hodnota);
 Serial.println(" ppm.");
 if (hodnota > 1000) {
   Serial.println("CO2 > 1000ppm, Moter On!!!");
   myStepper.step(stepsPerRevolution);
   delay(500);
 }
 delay(50);
}
