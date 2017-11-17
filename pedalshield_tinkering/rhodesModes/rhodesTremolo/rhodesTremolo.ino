// Licensed under a Creative Commons Attribution 3.0 Unported License.
// Based on rcarduino.blogspot.com previous work.
// www.electrosmash.com/pedalshield
 
float in_ADC0, in_ADC1;  //variables for 2 ADCs values (ADC0, ADC1)
int POT0, POT1, POT2, out_DAC0, out_DAC1; //variables for 3 pots (ADC8, ADC9, ADC10)
const int LED = 3;
const int FOOTSWITCH = 7; 
const int TOGGLE = 2; 

int tremRate;
int tremDepth;

int sample, accumulator, count, LFO;

// Create a table to hold pre computed sinewave, the table has a resolution of 600 samples
#define no_samples 44100
#define MAX_COUNT    160
uint16_t nSineTable[no_samples];//storing 12 bit samples in 16 bit variable.

// create the individual samples for our sinewave table
void createSineTable()
{
  for(uint32_t nIndex=0; nIndex<no_samples; nIndex++)
  {
    // normalised to 12 bit range 0-4095
    nSineTable[nIndex] = (uint16_t)  (((1+sin(((2.0*PI)/no_samples)*nIndex))*4095.0)/2);
  }
}


void setup()
{
  createSineTable();

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
  in_ADC0=(float) ADC->ADC_CDR[7];               // read data from ADC0
  in_ADC1=(float) ADC->ADC_CDR[6];               // read data from ADC1  
  POT0=ADC->ADC_CDR[10];                 // read data from ADC8        
  POT1=ADC->ADC_CDR[11];                 // read data from ADC9   
  POT2=ADC->ADC_CDR[12];                 // read data from ADC10     

  //Lowpass filter for noise
  in_ADC0 = ((in_ADC0*7)/100) + ((in_ADC0*93)/100);
  in_ADC1 = ((in_ADC1*7)/100) + ((in_ADC1*93)/100);
}


void TC4_Handler()
{
  
  TC_GetStatus(TC1, 1);

  ///////////////////
  //
  // read pots
  //
  
  tremRate = map(POT0, 0, 4095, 0, 2048);
  tremDepth = map(POT2, 0, 4095, 4095, 128);
  
  count++; 
  if (count>=160)
  {
    count=0;
    sample=sample+tremRate;
    if(sample>=no_samples) sample=0;
  }

  
  //sine wave tremolo.
  //LFO=map(nSineTable[sample],0,4095,tremDepth,4095);
  //out_DAC0 = map((int) in_ADC0,1,4095, 1, LFO);
  //out_DAC1 = map((int) in_ADC1,1,4095, 1, LFO);

  // square wave tremolo
  if (nSineTable[sample] > 2047) {
    out_DAC0 = map((int) in_ADC0,1, 4095, 1, tremDepth);
    out_DAC1 = map((int) in_ADC1,1, 4095, 1, tremDepth);    
  } else {
    out_DAC0 = (int) in_ADC0;
    out_DAC1 = (int) in_ADC1;
  }

  
  
  //Write the DACs
  dacc_set_channel_selection(DACC_INTERFACE, 0);          //select DAC channel 0
  dacc_write_conversion_data(DACC_INTERFACE, out_DAC0);   //write on DAC
  dacc_set_channel_selection(DACC_INTERFACE, 1);          //select DAC channel 1
  dacc_write_conversion_data(DACC_INTERFACE, out_DAC1);   //write on DAC
  
  //Turn on the LED if the effect is ON.
  if (digitalRead(FOOTSWITCH)) {
    if (nSineTable[sample] > 2047) {
      digitalWrite(LED, HIGH);
    } else {
      digitalWrite(LED, LOW);    
    }
  }
  
}
