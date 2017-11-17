// Licensed under a Creative Commons Attribution 3.0 Unported License.
// Based on rcarduino.blogspot.com previous work.
// www.electrosmash.com/pedalshield
 
int in_ADC0, in_ADC1;  //variables for 2 ADCs values (ADC0, ADC1)
double in_ADC0_filt = 2047;
double in_ADC1_filt = 2047;
int POT0, POT1, POT2, out_DAC0, out_DAC1; //variables for 3 pots (ADC8, ADC9, ADC10)
const int LED = 3;
const int FOOTSWITCH = 7; 
const int TOGGLE = 2; 

int nonlin = 100;

double filtStrength = 98; // higher = more filter applied
double filtParam0 = (100 - filtStrength) / filtStrength;
double filtParam1 = 100*filtStrength;

int loVal, hiVal;


void setup()
{
  //turn on the timer clock in the power management controller
  pmc_set_writeprotect(false);
  pmc_enable_periph_clk(ID_TC4);
 
  //we want wavesel 01 with RC 
  TC_Configure(TC1, 1, TC_CMR_WAVE | TC_CMR_WAVSEL_UP_RC | TC_CMR_TCCLKS_TIMER_CLOCK2);
  TC_SetRC(TC1, 1, 238); // sets <> 44.1 Khz interrupt rate
  TC_Start(TC1, 1);
 
  // enable timer interrupts on the timer
  TC1->TC_CHANNEL[1].TC_IER=TC_IER_CPCS;
  TC1->TC_CHANNEL[1].TC_IDR=~TC_IER_CPCS;
 
  //Enable the interrupt in the nested vector interrupt controller 
  //TC4_IRQn where 4 is the timer number * timer channels (3) + the channel 
  //number (=(1*3)+1) for timer1 channel1 
  NVIC_EnableIRQ(TC4_IRQn);
 
  //ADC Configuration
  ADC->ADC_MR |= 0x80;   // DAC in free running mode.
  ADC->ADC_CR=2;         // Starts ADC conversion.
  ADC->ADC_CHER=0x1CC0;  // Enable ADC channels 0,1,8,9 and 10  
 
  //DAC Configuration
  analogWrite(DAC0,0);  // Enables DAC0
  analogWrite(DAC1,0);  // Enables DAC1
  
  //pedalSHIELD pin configuration
  pinMode(LED, OUTPUT);  
  pinMode(FOOTSWITCH, INPUT_PULLUP);      
  pinMode(TOGGLE, INPUT_PULLUP);  
}
 
void loop()
{
  //Read the ADCs
  while((ADC->ADC_ISR & 0x1CC0)!=0x1CC0);// wait for ADC 0, 1, 8, 9, 10 conversion complete.
  in_ADC0= ADC->ADC_CDR[7];               // read data from ADC0
  in_ADC1= ADC->ADC_CDR[6];               // read data from ADC1  
  POT0=ADC->ADC_CDR[10];                 // read data from ADC8        
  POT1=ADC->ADC_CDR[11];                 // read data from ADC9   
  POT2=ADC->ADC_CDR[12];                 // read data from ADC10     

   //Lowpass filter for noise  
  in_ADC0_filt += double(in_ADC0)*filtParam0;
  in_ADC0_filt /= filtParam1;
  
  in_ADC1_filt += double(in_ADC1)*filtParam0;
  in_ADC1_filt /= filtParam1;

}

int sigmoid(double x, int nonlin)
{
  return (int) 4095.0 / ( 1.0 + exp((2047.0 - x) /  double(nonlin)) );
//  return int(x);
}


int twoSidedSqrt(double x, int nonlin)
{
  if (x > 2047) {
    return 2047 + int(sqrt(sqrt(x-2047)));
  } else {
    return 2047 - int(sqrt(sqrt(2047-x)));
  }
}

void TC4_Handler()
{
  
  TC_GetStatus(TC1, 1);

  ///////////////////
  //
  // read pots
  //
  
  nonlin = map(POT1, 0, 4095, 100, 1000);

  loVal = sigmoid(1024, nonlin);
  hiVal = sigmoid(4095-1024, nonlin);
  
  //out_DAC0 = twoSidedSqrt(in_ADC0_filt, nonlin);
  //out_DAC1 = twoSidedSqrt(in_ADC1_filt, nonlin);
  out_DAC0 = map( sigmoid(in_ADC0_filt, nonlin), loVal, hiVal, 0, 4095);
  out_DAC1 = map( sigmoid(in_ADC1_filt, nonlin), loVal, hiVal, 0, 4095);
  
  
  //Write the DACs
  dacc_set_channel_selection(DACC_INTERFACE, 0);          //select DAC channel 0
  dacc_write_conversion_data(DACC_INTERFACE, out_DAC0);   //write on DAC
  dacc_set_channel_selection(DACC_INTERFACE, 1);          //select DAC channel 1
  dacc_write_conversion_data(DACC_INTERFACE, out_DAC1);   //write on DAC
  
  //Turn on the LED if the effect is ON.
  if (digitalRead(FOOTSWITCH)) digitalWrite(LED, HIGH); 
  else  digitalWrite(LED, LOW);

}
