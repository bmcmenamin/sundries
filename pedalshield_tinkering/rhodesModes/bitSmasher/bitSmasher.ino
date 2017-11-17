// Licensed under a Creative Commons Attribution 3.0 Unported License.
// Based on rcarduino.blogspot.com previous work.
// www.electrosmash.com/pedalshield


//variables for 3 pots
int POT0, POT1, POT2;

//variables for 2 ADCs values
float in_ADC0, in_ADC1;

//variables for 2 DACs 
int  out_DAC0, out_DAC1;

// setting pin numbers
const int LED = 3;
const int FOOTSWITCH = 7; 
const int TOGGLE = 2; 

float lowpass_param = 10.0 / 100.0;
int input_upscaling = 1024;

// default values for quantization in amplitude and time
int reductionFactor_depth = 0;
int reductionFactor_time  = 0;

unsigned int DelayCounter = 0;




void setup()
{
  //turn on the timer clock in the power management controller
  pmc_set_writeprotect(false);
  pmc_enable_periph_clk(ID_TC4);
 
  //we want wavesel 01 with RC 
  TC_Configure(TC1, 1, TC_CMR_WAVE | TC_CMR_WAVSEL_UP_RC | TC_CMR_TCCLKS_TIMER_CLOCK2);
  
  // sets <> 44.1 Khz interrupt rate
  TC_SetRC(TC1, 1, 238);
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
  analogWrite(DAC0,0);
  analogWrite(DAC1,0);
  
  //pedalSHIELD pin configuration
  pinMode(LED,        OUTPUT);  
  pinMode(FOOTSWITCH, INPUT_PULLUP);      
  pinMode(TOGGLE,     INPUT_PULLUP);  
}


void loop()
{
  // wait for ADC 0, 1, 8, 9, 10 conversion complete.
  while((ADC->ADC_ISR & 0x1CC0)!=0x1CC0);

  //Read the ADCs as float, apply lowpass to help denoise
  in_ADC0 = in_ADC0*lowpass_param + ((float) input_upscaling*(ADC->ADC_CDR[7]))*(1-lowpass_param);
  in_ADC1 = in_ADC1*lowpass_param + ((float) input_upscaling*(ADC->ADC_CDR[6]))*(1-lowpass_param);
  
  //Read the pot values (ADC8-10)
  POT0=ADC->ADC_CDR[10];      
  POT1=ADC->ADC_CDR[11];  
  POT2=ADC->ADC_CDR[12];

  // map pots to values
  reductionFactor_time  = map(POT2, 0, 4095, 0, 9);
  reductionFactor_depth = map(POT0, 0, 4095, 10, 18);

}


void TC4_Handler()
{
  

  out_DAC0 += in_ADC0;
  out_DAC1 += in_ADC1;
   
  if(DelayCounter >= reductionFactor_time) {

    out_DAC0 /= reductionFactor_time + 1.0;
    out_DAC1 /= reductionFactor_time + 1.0;

    // map to low resolution, re-project back up

    out_DAC0 *= input_upscaling;
    out_DAC1 *= input_upscaling;
    out_DAC0 = ((int) out_DAC0 >> reductionFactor_depth) << reductionFactor_depth;
    out_DAC1 = ((int) out_DAC1 >> reductionFactor_depth) << reductionFactor_depth;
    out_DAC0 /= input_upscaling;
    out_DAC1 /= input_upscaling;

    //Write to the DACs
    dacc_set_channel_selection(DACC_INTERFACE, 0);          //select DAC channel 0
    dacc_write_conversion_data(DACC_INTERFACE, out_DAC0);   //write on DAC
    dacc_set_channel_selection(DACC_INTERFACE, 1);          //select DAC channel 1
    dacc_write_conversion_data(DACC_INTERFACE, out_DAC1);   //write on DAC
    
    DelayCounter = 0;
    out_DAC0 = 0;
    out_DAC1 = 0;
    
  }
  DelayCounter++;
    

  //Turn on the LED if the effect is ON.
  if (digitalRead(FOOTSWITCH)) digitalWrite(LED, HIGH); 
     else  digitalWrite(LED, LOW);
  
  TC_GetStatus(TC1, 1);

}
