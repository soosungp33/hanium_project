
#include "CO2Sensor.h"
CO2Sensor co2Sensor(A0, 0.99, 100);

void setup() {
  Serial.begin(9600);
  co2Sensor.calibrate();
}

void loop() {
  int hodnota = co2Sensor.read();
  Serial.print("차량 내부 CO2 농도 : ");
  Serial.print(hodnota);
  Serial.println(" ppm.");
  if (hodnota > 1000) {
    Serial.println("차량 내부의 CO2 농도가 1000ppm이 넘었습니다. 창문을 열어주세요!");
  }
  delay(1000);
}
