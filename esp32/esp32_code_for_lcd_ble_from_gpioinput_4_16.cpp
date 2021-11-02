//傳送是否有人接近4(有人接近回傳1)，透過esp32分別傳給顯示器及手機。
//傳送是否有人接近16(有人接近回傳1)，透過esp32分別傳給顯示器及手機。
#include <Wire.h> 
#include <LiquidCrystal_I2C.h>
LiquidCrystal_I2C lcd(0x27,16,2);//設定LCD位址與大小

#include <BluetoothSerial.h>
BluetoothSerial BT;//宣告藍芽物件，名稱為BT  //BT為藍芽通訊物件

int ble_count=0;//紀錄esp32用ble通訊傳了幾次訊息到手機的android APP
int flag=0;
//int insert=0;  //from 0-3 repeat, use to check if the people is still standing here.
//int still[4]={0};  // use to check if the people is still standing here.  once still={1,1,1,1}  //1 mean people come
int car_count=0; //目前內部的車子數量
int max_park=30;

void setup()
{
  lcd.begin();   //初始化LCD
  lcd.backlight(); //開啟LCD背光

  pinMode(4, INPUT); //宣告GPIO 4作為數位輸入  //超音波測距感測器1 //測有無人接近
  pinMode(16, INPUT); //宣告GPIO 16作為數位輸入  //超音波測距感測器2 //測有無人接近

  Serial.begin(115200);
  BT.begin("anderson_0001");//修改名稱為自己的廣播名稱(可隨意)  //在Setup中開啟藍芽  // BTName為藍芽廣播名稱
  delay(3000);
}

//藍芽單向傳輸
//使用ESP32藍芽將資訊傳遞給藍芽裝置（手機）進行接收
//顯示剩餘車位
void ble_to_android_phone(void) {
    if(car_count>=max_park){
        BT.print(ble_count);
        BT.print(" time: ");
        BT.print("NOT Available!!");//沒位子
    }
    else{
        BT.print(ble_count);
        BT.print(" time: ");
        BT.print("now we have ");
        BT.print(max_park - car_count); //顯示剩餘車位
        BT.println(" spaces available");
        //BT.println("Wish you have a good day! :))");
    }
    ble_count++;
    delay(1000);//這個delay不能太小，不然來不及傳出去
}

void coming(void){//控制insert  car_count  flag  still[] 的主軸
    if (digitalRead(4) == HIGH  || digitalRead(16) == HIGH) {
        //still[insert]=1;
        car_count++;
        flag=1;
        //insert++;
        //if(insert==4)
            //insert=0;
    }
    else flag=0;
}

//void stand(void){
//  int m=1;
//  for(int i=0;i<4;i++){
//      if(still[i]==0)//只有有任何一次沒人靠近，m=0
//        m=still[i];
//  }
//
//  if(m==1){//表示最近4次都有人靠近
//    lcd.setCursor(0,0);//設定游標
//    lcd.println("COMING for a while!!");
//    delay(300);
//    lcd.clear();//清除所有內容
//    delay(300);
//  }
//}

void esp_i2c_lcd(void){
  
  lcd.setCursor(0,0);//設定游標
  if(flag==1){
    lcd.println("SOMEBODY COMING!!");
    delay(600);
    //lcd.clear();//清除所有內容
   }
  else{
    lcd.println("NOBODY COMING!!"); 
    delay(600);
    //lcd.clear();//清除所有內容
  }
}

void loop(){
  coming();//由imput設定flag
  ble_to_android_phone(); //顯示剩餘車位
  esp_i2c_lcd(); //顯示剩餘車位
}
    