import random
import re
from typing import Dict, Any, List, Optional  


class TextPool:
    def __init__(self):
        self.ts_characteristic_desc_pool = {
            "seasonality": {
                # No seasonal pattern combinations
                ("no periodic fluctuation", "no periodicity"): [
                    "The time series does not exhibit any periodic fluctuations or seasonal patterns.",
                    "There are no discernible periodic patterns or seasonality in the time series.",
                    "The time series appears to be non-seasonal and non-periodic, showing no regular cycles.",
                    "No periodic fluctuations or seasonal patterns are observed in the time series.",
                    "No repeating cycles or seasonal signatures are present in the data.",
                    "The series lacks any stable periodic components across the observed range.",
                    "No cyclic structure is detected; behavior remains aperiodic.",
                    "No evidence of recurring intervals or seasonal repetition is found.",
                    "The profile is aperiodic with no regular time-based recurrence.",
                    "Seasonal effects are absent; the signal shows no rhythmic repetition.",
                    "The sequence does not reveal periodic recurrence or seasonality cues.",
                    "No regular oscillations occur; patterns do not repeat over fixed intervals.",
                    "There is no clocked cadence; the series behaves without periodic order.",
                    "An absence of seasonal patterns is noted across the timeline.",
                    "No persistent cycle length or periodic rhythm can be identified.",
                    "The data remains non-cyclical throughout the observation period."
                ],
                
                # Irregular pattern (no specific frequency)
                ("irregular pattern", "no periodicity"): [
                    "The time series exhibits an irregular pattern with no consistent periodicity, where patterns appear randomly.",
                    "An irregular pattern is present without regular frequency, characterized by unpredictable timing of similar shapes.",
                    "The time series shows an irregular pattern that lacks consistent periodicity, appearing at random intervals.",
                    "A fixed-length pattern appears irregularly with no specific frequency in the time series.",
                    "Irregular motifs recur sporadically without a stable cadence or cycle.",
                    "The sequence contains intermittent shapes that do not align to a regular period.",
                    "Pattern occurrences are scattered, lacking any fixed timing.",
                    "Recurring segments surface unpredictably without a reliable interval.",
                    "No fixed frequency governs the irregular repetitions observed.",
                    "Uneven recurrence yields irregular pattern appearances across time.",
                    "Similar shapes reappear in a non-periodic, random manner.",
                    "Irregular fragments emerge inconsistently without synchronized timing.",
                    "The data shows sporadic template matches absent of periodic order.",
                    "Irregular repeats exist but do not follow a clocked schedule."
                ],
                
                # Sinusoidal patterns with different frequencies
                ("sin periodic fluctuation", "minutely"): [
                    "The time series exhibits smooth sinusoidal fluctuations with minute-level periodicity.",
                    "A sinusoidal pattern repeats every few minutes, creating smooth periodic variations.",
                    "The time series shows minute-level sinusoidal cycles, indicating high-frequency smooth seasonal patterns.",
                    "Smooth sinusoidal periodic patterns occur at minute-level intervals in the time series.",
                    "Minute-resolution cycles follow a gentle sinusoidal waveform.",
                    "Regular minute-scale oscillations form a clean sinus rhythm.",
                    "Wave-like minute patterns recur with smooth up-and-down motion.",
                    "Consistent minute-period sine waves shape the seasonal behavior.",
                    "A soft sinus curve repeats on a minute cadence across the series.",
                    "Minute-by-minute oscillations manifest as a stable sinus track.",
                    "High-frequency minute cycles present a clear sinus structure.",
                    "Sustained minute cycles trace a sinusoidal envelope.",
                    "Minute periodicity is evident through smooth sine oscillations.",
                    "The minute grid reveals repeating sinus peaks and troughs."
                ],
                ("sin periodic fluctuation", "hourly"): [
                    "The time series demonstrates hourly sinusoidal periodic fluctuations with smooth transitions.",
                    "A sinusoidal pattern repeats every hour, showing regular smooth seasonal variations.",
                    "The time series exhibits hourly sinusoidal cycles, reflecting consistent smooth periodic patterns.",
                    "Smooth sinusoidal fluctuations occur with hourly periodicity throughout the time series.",
                    "Hourly oscillations follow a steady sinus wave profile.",
                    "Each hour brings a repeating sinus cycle with gentle curvature.",
                    "A well-formed hourly sine rhythm persists across the timeline.",
                    "Hourly cadence produces predictable sinus peaks and troughs.",
                    "Clear hourly periodicity manifests as smooth sine-like motion.",
                    "The signal tracks an hourly sinusoid with consistent magnitude.",
                    "Regular hourly cycles trace a sinus envelope without abrupt changes.",
                    "Hourly sine patterns recur stably across observations.",
                    "An hourly sinusoidal signature governs seasonal variation.",
                    "The hourly grid highlights repeating smooth sinus fluctuations."
                ],
                ("sin periodic fluctuation", "daily"): [
                    "The time series shows daily sinusoidal periodic fluctuations, indicating smooth day-to-day patterns.",
                    "A sinusoidal pattern repeats daily, creating smooth and consistent seasonal variations.",
                    "The time series exhibits daily sinusoidal cycles, reflecting smooth periodic daily patterns.",
                    "Smooth sinusoidal fluctuations with daily periodicity are observed in the time series.",
                    "A steady daily sine rhythm recurs with predictable regularity.",
                    "Daily cycles trace a smooth sinus curve over successive days.",
                    "Gentle day-scale oscillations repeat with sinusoidal form.",
                    "Daily periodicity is governed by consistent sinus transitions.",
                    "Day-by-day sine patterns appear with stable magnitude and phase.",
                    "The series follows a daily sinus waveform without sharp breaks.",
                    "Daily sinus peaks and troughs recur across the observation horizon.",
                    "A coherent daily sinus signature defines the seasonal behavior.",
                    "Daily sinusoidal motion remains smooth and uniform throughout.",
                    "Regular day-length sine cycles structure the time series pattern."
                ],
                ("sin periodic fluctuation", "weekly"): [
                    "The time series demonstrates weekly sinusoidal periodic fluctuations with smooth patterns.",
                    "A sinusoidal pattern repeats weekly, showing smooth and regular weekly seasonal variations.",
                    "The time series exhibits weekly sinusoidal cycles, indicating smooth weekly periodic patterns.",
                    "Smooth sinusoidal fluctuations occur with weekly periodicity throughout the time series.",
                    "Weekly oscillations present a stable sinus profile across weeks.",
                    "A consistent week-scale sine rhythm underpins the seasonal shape.",
                    "Week-by-week sinus cycles recur with steady magnitude.",
                    "Weekly periodicity emerges as smooth sine waves over the calendar.",
                    "The signal displays a weekly sinus envelope without discontinuities.",
                    "Weekly sine peaks and troughs align with a regular cadence.",
                    "Coherent weekly sinusoidal motion persists through the dataset.",
                    "Weekly cycles trace gentle sinus arcs repeatedly.",
                    "A weekly sinus signature characterizes the recurring pattern.",
                    "Stable weekly sine oscillations structure the time series."
                ],
                
                # Square wave patterns with different frequencies
                ("square periodic fluctuation", "minutely"): [
                    "The time series exhibits square wave fluctuations with minute-level periodicity, showing abrupt transitions.",
                    "A square wave pattern repeats every few minutes, creating sharp periodic changes.",
                    "The time series shows minute-level square wave cycles with abrupt shifts between levels.",
                    "Sharp square wave periodic patterns occur at minute-level intervals in the time series.",
                    "Minute-scale square pulses produce abrupt high-low transitions.",
                    "Clear minute cadence yields blocky on/off level shifts.",
                    "Regular minute ticks drive step-like square oscillations.",
                    "A crisp minute waveform alternates sharply between states.",
                    "Discrete minute intervals create rectangular periodic swings.",
                    "Minute-resolution steps form a repeatable square pattern.",
                    "Abrupt minute transitions define a square-shaped cycle.",
                    "Stable minute blocks repeat with sharp edges and flat tops.",
                    "A minute square rhythm features sudden plateaus and drops.",
                    "Consistent minute square waves exhibit binary-like level changes."
                ],
                ("square periodic fluctuation", "hourly"): [
                    "The time series demonstrates hourly square wave periodic fluctuations with abrupt changes.",
                    "A square wave pattern repeats every hour, showing sharp transitions in seasonal patterns.",
                    "The time series exhibits hourly square wave cycles with sudden shifts between states.",
                    "Abrupt square wave fluctuations occur with hourly periodicity throughout the time series.",
                    "Hourly step functions alternate states with sharp edges.",
                    "A blocky hourly waveform switches abruptly between levels.",
                    "Regular hourly squares show distinct plateaus and jumps.",
                    "Hour-by-hour, the signal toggles between discrete states.",
                    "The hourly cadence produces rectangular pulses with crisp transitions.",
                    "Hourly square cycles maintain flat segments followed by sudden changes.",
                    "Sharp hourly toggles define the square periodic signature.",
                    "A repeating hourly square profile emerges without smoothing.",
                    "Consistent hourly block transitions characterize the pattern.",
                    "Hourly square oscillations are evident across the timeline."
                ],
                ("square periodic fluctuation", "daily"): [
                    "The time series shows daily square wave periodic fluctuations, indicating abrupt day-to-day changes.",
                    "A square wave pattern repeats daily, creating sharp and sudden seasonal transitions.",
                    "The time series exhibits daily square wave cycles with abrupt shifts between daily patterns.",
                    "Sharp square wave fluctuations with daily periodicity are observed in the time series.",
                    "A day-scale square rhythm alternates distinctly across days.",
                    "Daily blocks form rectangular levels with sudden step changes.",
                    "Day-by-day toggling between flat regimes delineates square behavior.",
                    "The daily cadence produces clear plateaus and drops in a square form.",
                    "Consistent daily square pulses bring abrupt daily transitions.",
                    "Square-shaped daily oscillations persist with crisp boundaries.",
                    "Daily squares repeat with binary-like day-level changes.",
                    "Rectangular daily cycles appear as sharp state switches.",
                    "Daily square wave signatures stay prominent through the series.",
                    "Clear day-level step shifts define the periodic pattern."
                ],
                ("square periodic fluctuation", "weekly"): [
                    "The time series demonstrates weekly square wave periodic fluctuations with sharp transitions.",
                    "A square wave pattern repeats weekly, showing abrupt changes in weekly seasonal patterns.",
                    "The time series exhibits weekly square wave cycles with sudden shifts between weekly states.",
                    "Abrupt square wave fluctuations occur with weekly periodicity throughout the time series.",
                    "Weekly rectangular pulses switch regimes with steep edges.",
                    "Week-by-week, flat segments alternate in a square fashion.",
                    "A blocky weekly profile features clear plateaus and jumps.",
                    "Weekly step transitions produce a distinct square cadence.",
                    "Square-like weekly oscillations manifest as on/off state changes.",
                    "Weekly squares recur with crisp boundaries and steady levels.",
                    "Discrete weekly toggles define the periodic shape.",
                    "Sharp weekly regime changes repeat with square regularity.",
                    "Rectangular week-scale cycles are evident across the record.",
                    "Weekly square wave behavior persists throughout the timeline."
                ],
                
                # Triangular patterns with different frequencies
                ("triangle periodic fluctuation", "minutely"): [
                    "The time series exhibits triangular fluctuations with minute-level periodicity, showing linear rise and fall patterns.",
                    "A triangular pattern repeats every few minutes, creating consistent linear periodic changes.",
                    "The time series shows minute-level triangular cycles with steady rise and fall transitions.",
                    "Linear triangular periodic patterns occur at minute-level intervals in the time series.",
                    "Minute-scale ramps ascend and descend in a triangular form.",
                    "Regular minute triangles present linear up-then-down motion.",
                    "The minute cadence yields symmetric triangular slopes.",
                    "Minute-period sawtooth-like triangles recur with steady gradients.",
                    "Consistent minute triangles show linear segments without curvature.",
                    "Minute triangles alternate between rising and falling lines.",
                    "Stable minute ramp-up and ramp-down cycles shape the triangles.",
                    "A minute triangle waveform appears with clean linear edges.",
                    "Minute triangular oscillations persist throughout the series.",
                    "Linear minute slopes form repeating triangular cycles."
                ],
                ("triangle periodic fluctuation", "hourly"): [
                    "The time series demonstrates hourly triangular periodic fluctuations with linear transitions.",
                    "A triangular pattern repeats every hour, showing steady rise and fall in seasonal patterns.",
                    "The time series exhibits hourly triangular cycles with consistent linear changes.",
                    "Linear triangular fluctuations occur with hourly periodicity throughout the time series.",
                    "Hourly triangles trace alternating linear ascents and descents.",
                    "A clear hourly ramp pattern forms symmetric triangular shapes.",
                    "Regular hourly slopes create a repeating triangular profile.",
                    "Hour-by-hour, linear segments compose a triangular rhythm.",
                    "Triangular hourly cycles maintain steady gradient changes.",
                    "The hourly timeline features sawtooth-like triangular sequences.",
                    "Hourly ramp cycles alternate in a triangle configuration.",
                    "Consistent hourly triangles show clean straight-line transitions.",
                    "A stable hourly triangle motif recurs across the observation window.",
                    "Linear hourly up-and-down motions define the periodicity."
                ],
                ("triangle periodic fluctuation", "daily"): [
                    "The time series shows daily triangular periodic fluctuations, indicating linear day-to-day patterns.",
                    "A triangular pattern repeats daily, creating steady rise and fall seasonal variations.",
                    "The time series exhibits daily triangular cycles with consistent linear daily patterns.",
                    "Linear triangular fluctuations with daily periodicity are observed in the time series.",
                    "Daily ramps form symmetric triangular ascents and descents.",
                    "A day-scale triangle waveform repeats with straight-line segments.",
                    "Daily slopes alternate to build a repeating triangular structure.",
                    "Day-by-day triangular cycles maintain linear transitions.",
                    "Consistent daily triangles trace clean linear edges.",
                    "Daily triangular patterns persist with uniform gradient changes.",
                    "Triangular day-length cycles show steady rise and fall lines.",
                    "A stable daily triangle rhythm appears across days.",
                    "Daily sawtooth-like triangles recur without curvature.",
                    "Linear daily up/down segments define the triangular periodicity."
                ],
                ("triangle periodic fluctuation", "weekly"): [
                    "The time series demonstrates weekly triangular periodic fluctuations with linear patterns.",
                    "A triangular pattern repeats weekly, showing steady rise and fall in weekly seasonal patterns.",
                    "The time series exhibits weekly triangular cycles with consistent linear weekly changes.",
                    "Linear triangular fluctuations occur with weekly periodicity throughout the time series.",
                    "Weekly triangles alternate linear increases and decreases consistently.",
                    "A weekly triangle motif emerges as straight-line ramps.",
                    "Week-by-week triangular cycles present uniform slopes.",
                    "Triangular weekly rhythms show clean linear transitions.",
                    "Consistent weekly triangular sequences repeat across the record.",
                    "Weekly sawtooth triangles persist with stable gradients.",
                    "The weekly timeline features symmetric triangular patterns.",
                    "Weekly ramps up and down compose a triangular periodic shape.",
                    "Straight weekly segments define recurring triangular cycles.",
                    "Linear weekly ascents/descents characterize the periodicity."
                ]
            },
            "trend": {
                "decrease": [
                    "The time series exhibits a decreasing trend, indicating a downward movement over time.",
                    "A downward trend is observed in the time series, suggesting a consistent decrease over time.",
                    "The time series shows a decreasing trend, reflecting a decline over the observed period.",
                    "A consistent downward movement is evident in the time series, indicating a decreasing trend.",
                    "Values drift lower over time, forming a clear downward trajectory.",
                    "The series gradually declines, marking a persistent negative slope.",
                    "A sustained drop is visible, pointing to long-run weakening.",
                    "Measurements trend downward with diminishing levels across the window.",
                    "A negative trend is present, with progressive reductions.",
                    "Overall direction points down, with stepwise decreases over time.",
                    "The baseline slides downward, indicating contraction.",
                    "A steady fall pattern emerges in the time series.",
                    "Downward pressure persists, lowering values as time advances.",
                    "A weakening trajectory is maintained throughout the period."
                ],
                "increase": [
                    "The time series demonstrates an increasing trend, indicating an upward movement over time.",
                    "An upward trend is observed in the time series, suggesting a consistent increase over time.",
                    "The time series shows an increasing trend, reflecting growth over the observed period.",
                    "A consistent upward movement is evident in the time series, indicating an increasing trend.",
                    "Values climb over time, tracing a clear upward trajectory.",
                    "The series gradually rises, revealing a persistent positive slope.",
                    "A sustained lift is visible, pointing to long-run strengthening.",
                    "Measurements trend upward with increasing levels across the window.",
                    "A positive trend is present, with progressive gains.",
                    "Overall direction points up, with stepwise increases over time.",
                    "The baseline shifts upward, indicating expansion.",
                    "A steady rise pattern emerges in the time series.",
                    "Upward momentum persists, elevating values as time advances.",
                    "A strengthening trajectory is maintained throughout the period."
                ],
                "keep steady": [
                    "The time series maintains a steady trend, indicating stability over time.",
                    "A stable trend is observed in the time series, suggesting little to no change over time.",
                    "The time series shows a steady trend, reflecting consistency over the observed period.",
                    "Little to no change is evident in the time series, indicating a steady trend.",
                    "Values remain relatively flat, with negligible drift.",
                    "The baseline is stable, showing no persistent direction.",
                    "No pronounced up or down movement; the series is steady.",
                    "A neutral trajectory holds with minimal variation.",
                    "The overall level is consistent across time, without trend.",
                    "Signal stays near a constant band, lacking directional bias.",
                    "A flat profile dominates, indicating equilibrium.",
                    "Trend-less behavior persists, maintaining stability.",
                    "The series hovers around a stable mean without slope.",
                    "No sustained drift detected; the trend is steady."
                ],
                "random": [
                    "The time series exhibits a random fluctuation trend pattern, showing irregular movements without a clear directional trend.",
                    "The time series displays irregular oscillations, suggesting a lack of consistent trend direction.",
                    "An erratic pattern is evident in the time series, characterized by random ups and downs without a clear trend.",
                    "Irregular variations are present in the time series, indicating a random and unpredictable trend behavior.",
                    "The time series demonstrates a stochastic pattern with no discernible directional trend.",
                    "Movements are noisy and directionless, lacking persistent slope.",
                    "Volatility dominates with no consistent drift up or down.",
                    "Random walk-like fluctuations obscure any trend signal.",
                    "The path meanders unpredictably without a stable direction.",
                    "Chaotic oscillations occur, offering no clear trajectory.",
                    "The series wanders with unstructured rises and falls.",
                    "No coherent trend emerges; variability is predominant.",
                    "Direction changes frequently, resulting in non-trending behavior.",
                    "A pattern of erratic swings prevents trend identification."
                ]
            },
            "noise": {
                "noisy": [
                    "The time series contains a significant amount of noise, which may obscure underlying patterns.",
                    "A high level of noise is present in the time series, potentially complicating analysis.",
                    "The time series is characterized by considerable noise, making it challenging to discern trends.",
                    "Significant noise in the time series may hinder the identification of underlying patterns.",
                    "Noise levels are elevated, reducing signal clarity.",
                    "Substantial random variation masks structure in the data.",
                    "High-frequency disturbances blur the underlying dynamics.",
                    "Pronounced noise complicates detection of seasonal or trend components.",
                    "The signal-to-noise ratio is low, making interpretation difficult.",
                    "Noisy fluctuations dominate, obscuring consistent behavior.",
                    "Irregular jitter is prevalent across the series.",
                    "Broad variance from noise interferes with pattern recognition.",
                    "Residuals appear large, indicating heavy noise contamination.",
                    "Considerable noise persists throughout the observation period."
                ],
                "almost no noise": [
                    "The time series is relatively clean, with minimal noise allowing for clear pattern recognition.",
                    "A low level of noise is present in the time series, facilitating easier analysis of trends.",
                    "The time series exhibits minimal noise, making it straightforward to identify underlying patterns.",
                    "With almost no noise, the time series allows for clear detection of trends and patterns.",
                    "Noise is minimal; the signal remains crisp and interpretable.",
                    "Low noise improves visibility of seasonal and trend structures.",
                    "Variability from noise is small, enhancing pattern clarity.",
                    "A clean signal permits reliable characterization of behavior.",
                    "Only slight random disturbances are observed over time.",
                    "The series is quiet, with near-negligible noise components.",
                    "Stable readings show little contamination from random noise.",
                    "High signal-to-noise ratio supports precise analysis.",
                    "Minimal jitter is present, keeping dynamics easy to track.",
                    "Noise contributions are minor across the dataset."
                ]
            }
        }

        self.anomaly_pattern_desc_pool = {
            "spike": [
                "Spike anomaly: {duration} points forming spike shape with magnitude {magnitude:.3g} from position {start_pos} to {end_pos}.",
                "A spike anomaly is detected, characterized by a sharp peak of magnitude {magnitude:.3g} over {duration} points between positions {start_pos} and {end_pos}.",
                "The time series shows a spike anomaly, with a sudden increase and decrease (magnitude {magnitude:.3g}) over {duration} points from position {start_pos} to {end_pos}.",
                "An abrupt spike pattern with magnitude {magnitude:.3g} occurs over {duration} points, spanning from position {start_pos} to {end_pos}.",
                "A short-lived surge forms a spike (magnitude {magnitude:.3g}) within {duration} points from {start_pos} to {end_pos}.",
                "Sudden peak: magnitude {magnitude:.3g}, lasting {duration} points between {start_pos} and {end_pos}.",
                "A transient spike rises and falls quickly (magnitude {magnitude:.3g}) across {duration} points [{start_pos}, {end_pos}].",
                "Sharp spike observed with magnitude {magnitude:.3g} over {duration} samples from {start_pos} to {end_pos}.",
                "Brief peak anomaly reaching {magnitude:.3g} magnitude, spanning {duration} points ({start_pos}–{end_pos}).",
                "Instantaneous spike manifests (magnitude {magnitude:.3g}) across {duration} points, positions {start_pos}–{end_pos}.",
                "A pronounced spike episode (magnitude {magnitude:.3g}) persists for {duration} points between {start_pos} and {end_pos}.",
                "Single spike event: magnitude {magnitude:.3g}, window {duration} points [{start_pos}, {end_pos}].",
                "A fast peak and decay (magnitude {magnitude:.3g}) detected across {duration} points from {start_pos} to {end_pos}.",
                "Spike signature: sharp rise/fall with magnitude {magnitude:.3g} over {duration} points ({start_pos}–{end_pos})."
            ],
            "dip": [
                "Dip anomaly: {duration} points forming dip shape with magnitude {magnitude:.3g} from position {start_pos} to {end_pos}.",
                "A dip anomaly is identified, marked by a sharp trough of magnitude {magnitude:.3g} over {duration} points between positions {start_pos} and {end_pos}.",
                "The time series exhibits a dip anomaly, with a sudden decrease and recovery (magnitude {magnitude:.3g}) over {duration} points from position {start_pos} to {end_pos}.",
                "An abrupt dip pattern with magnitude {magnitude:.3g} occurs over {duration} points, spanning from position {start_pos} to {end_pos}.",
                "Sudden trough detected: magnitude {magnitude:.3g}, lasting {duration} points between {start_pos} and {end_pos}.",
                "Transient dip: a quick drop and recovery (magnitude {magnitude:.3g}) across {duration} points [{start_pos}, {end_pos}].",
                "A short-lived valley forms (magnitude {magnitude:.3g}) within {duration} points, positions {start_pos}–{end_pos}.",
                "Pronounced dip anomaly of magnitude {magnitude:.3g} over {duration} samples from {start_pos} to {end_pos}.",
                "Brief trough event reaching {magnitude:.3g} magnitude, spanning {duration} points ({start_pos}–{end_pos}).",
                "Instantaneous downturn (magnitude {magnitude:.3g}) detected across {duration} points from {start_pos} to {end_pos}.",
                "A deep dip episode (magnitude {magnitude:.3g}) persists for {duration} points between {start_pos} and {end_pos}.",
                "Single trough event: magnitude {magnitude:.3g}, window {duration} points [{start_pos}, {end_pos}].",
                "Dip signature: sharp fall/rise with magnitude {magnitude:.3g} over {duration} points ({start_pos}–{end_pos})."
            ],
            "multi_spike": [
                "Multiple spike anomaly: {duration} points containing multiple spike patterns with magnitude {magnitude:.3g} from position {start_pos} to {end_pos}.",
                "A multiple spike anomaly is observed, featuring several sharp peaks with typical magnitude {magnitude:.3g} over {duration} points between positions {start_pos} and {end_pos}.",
                "The time series shows multiple spike anomalies, with several sudden increases and decreases (magnitude {magnitude:.3g}) over {duration} points from position {start_pos} to {end_pos}.",
                "Several abrupt spike patterns with magnitude {magnitude:.3g} occur over {duration} points, spanning from position {start_pos} to {end_pos}.",
                "A cluster of spike events (typical magnitude {magnitude:.3g}) emerges across {duration} points [{start_pos}, {end_pos}].",
                "Multiple short peaks appear, each around magnitude {magnitude:.3g}, within {duration} points from {start_pos} to {end_pos}.",
                "Repeated spike signatures (magnitude ~{magnitude:.3g}) are concentrated over {duration} points ({start_pos}–{end_pos}).",
                "Several rapid peak-and-decay episodes (mag {magnitude:.3g}) occur between {start_pos} and {end_pos}.",
                "Multiple spike-like surges (magnitude {magnitude:.3g}) populate the range {start_pos}–{end_pos} over {duration} points.",
                "A sequence of spikes repeats with magnitude near {magnitude:.3g} across {duration} points between {start_pos} and {end_pos}.",
                "Dense spike activity (mag {magnitude:.3g}) spans {duration} samples from {start_pos} to {end_pos}.",
                "Multispike pattern: several peaks with magnitude {magnitude:.3g} within {duration} points ({start_pos}–{end_pos})."
            ],
            "multi_dip": [
                "Multiple dip anomaly: {duration} points containing multiple dip patterns with magnitude {magnitude:.3g} from position {start_pos} to {end_pos}.",
                "A multiple dip anomaly is detected, characterized by several sharp troughs with typical magnitude {magnitude:.3g} over {duration} points between positions {start_pos} and {end_pos}.",
                "The time series exhibits multiple dip anomalies, with several sudden decreases and recoveries (magnitude {magnitude:.3g}) over {duration} points from position {start_pos} to {end_pos}.",
                "Several abrupt dip patterns with magnitude {magnitude:.3g} occur over {duration} points, spanning from position {start_pos} to {end_pos}.",
                "A cluster of dip events (typical magnitude {magnitude:.3g}) emerges across {duration} points [{start_pos}, {end_pos}].",
                "Multiple short troughs appear, each around magnitude {magnitude:.3g}, within {duration} points from {start_pos} to {end_pos}.",
                "Repeated dip signatures (magnitude ~{magnitude:.3g}) are concentrated over {duration} points ({start_pos}–{end_pos}).",
                "Several rapid fall-and-recovery episodes (mag {magnitude:.3g}) occur between {start_pos} and {end_pos}.",
                "Multiple dip-like downturns (magnitude {magnitude:.3g}) populate the range {start_pos}–{end_pos} over {duration} points.",
                "A sequence of dips repeats with magnitude near {magnitude:.3g} across {duration} points from {start_pos} to {end_pos}.",
                "Dense dip activity (mag {magnitude:.3g}) spans {duration} samples from {start_pos} to {end_pos}.",
                "Multidip pattern: several troughs with magnitude {magnitude:.3g} within {duration} points ({start_pos}–{end_pos})."
            ],
            "sudden_increase": [
                "Sudden increase anomaly: rapid rise to a new level with magnitude {magnitude:.3g} then stabilize from position {start_pos} to {end_pos}.",
                "A sudden increase anomaly is detected, marked by a rapid upward shift of magnitude {magnitude:.3g} over {duration} points between positions {start_pos} and {end_pos}.",
                "The time series exhibits a sudden increase anomaly, with an abrupt rise to a new stable level (magnitude {magnitude:.3g}) over {duration} points from position {start_pos} to {end_pos}.",
                "An abrupt upward level shift of magnitude {magnitude:.3g} occurs over {duration} points, spanning from position {start_pos} to {end_pos}.",
                "Rapid uplift to a higher regime (magnitude {magnitude:.3g}) within {duration} points from {start_pos} to {end_pos}.",
                "A step-like jump upward (magnitude {magnitude:.3g}) is completed across {duration} points [{start_pos}, {end_pos}].",
                "Instant level increase with magnitude {magnitude:.3g} over {duration} points between {start_pos} and {end_pos}.",
                "Abrupt upward shift transitions the series to a new plateau (mag {magnitude:.3g}) in {duration} points ({start_pos}–{end_pos}).",
                "A sharp rise event (magnitude {magnitude:.3g}) occurs across {duration} samples from {start_pos} to {end_pos}.",
                "Level-change anomaly: upward jump of {magnitude:.3g} within {duration} points ({start_pos}–{end_pos}).",
                "A quick increase to a new stable band (magnitude {magnitude:.3g}) from {start_pos} to {end_pos}."
            ],
            "sudden_decrease": [
                "Sudden decrease anomaly: rapid drop to a new level with magnitude {magnitude:.3g} then stabilize from position {start_pos} to {end_pos}.",
                "A sudden decrease anomaly is identified, characterized by a rapid downward shift of magnitude {magnitude:.3g} over {duration} points between positions {start_pos} and {end_pos}.",
                "The time series shows a sudden decrease anomaly, with an abrupt drop to a new stable level (magnitude {magnitude:.3g}) over {duration} points from position {start_pos} to {end_pos}.",
                "An abrupt downward level shift of magnitude {magnitude:.3g} occurs over {duration} points, spanning from position {start_pos} to {end_pos}.",
                "Rapid decline to a lower regime (magnitude {magnitude:.3g}) within {duration} points from {start_pos} to {end_pos}.",
                "A step-like drop (magnitude {magnitude:.3g}) is completed across {duration} points [{start_pos}, {end_pos}].",
                "Instant level decrease with magnitude {magnitude:.3g} over {duration} points between {start_pos} and {end_pos}.",
                "A sharp fall event (magnitude {magnitude:.3g}) occurs across {duration} samples from {start_pos} to {end_pos}.",
                "Level-change anomaly: downward jump of {magnitude:.3g} within {duration} points ({start_pos}–{end_pos}).",
                "A quick decrease to a new stable band (magnitude {magnitude:.3g}) from {start_pos} to {end_pos}."
            ],
            "slow_increase": [
                "Slow increase anomaly: gradual upward trend with total magnitude {magnitude:.3g} over {duration} points from position {start_pos} to {end_pos}.",
                "A slow increase anomaly is detected, marked by a gradual upward drift of magnitude {magnitude:.3g} over {duration} points between positions {start_pos} and {end_pos}.",
                "The time series exhibits a slow increase anomaly, with a steady upward trend (magnitude {magnitude:.3g}) over {duration} points from position {start_pos} to {end_pos}.",
                "A gradual upward drift pattern of magnitude {magnitude:.3g} occurs over {duration} points, spanning from position {start_pos} to {end_pos}.",
                "Gradual rise anomaly: cumulative change {magnitude:.3g} across {duration} points ({start_pos}–{end_pos}).",
                "A slow upward drift accumulates (magnitude {magnitude:.3g}) over {duration} points, from position {start_pos} to {end_pos}.",
                "Progressive increase towards a higher level (total magnitude {magnitude:.3g}) in {duration} points ({start_pos}–{end_pos}).",
                "Soft ramp-up pattern with net change {magnitude:.3g} across {duration} points between {start_pos} to {end_pos}.",
                "A gentle climb unfolds over {duration} points between {start_pos} to {end_pos}, magnitude {magnitude:.3g}.",
                "Slow escalation trend observed with magnitude {magnitude:.3g} over {duration} points,  from {start_pos} to {end_pos}.",
                "A gradual level uplift (magnitude {magnitude:.3g}) from {start_pos} to {end_pos}.",
            ],
            "slow_decrease": [
                "Slow decrease anomaly: gradual downward trend with total magnitude {magnitude:.3g} over {duration} points from position {start_pos} to {end_pos}.",
                "A slow decrease anomaly is identified, characterized by a gradual downward drift of magnitude {magnitude:.3g} over {duration} points between positions {start_pos} and {end_pos}.",
                "The time series shows a slow decrease anomaly, with a steady downward trend (magnitude {magnitude:.3g}) over {duration} points from position {start_pos} to {end_pos}.",
                "A gradual downward drift pattern of magnitude {magnitude:.3g} occurs over {duration} points, spanning from position {start_pos} to {end_pos}.",
                "Gradual decline anomaly: cumulative change {magnitude:.3g} across {duration} points ({start_pos}–{end_pos}).",
                "A slow downward drift accumulates (magnitude {magnitude:.3g}) over {duration} points from {start_pos} to {end_pos}.",
                "Progressive decrease towards a lower level (total magnitude {magnitude:.3g}) in {duration} points ({start_pos}–{end_pos}).",
                "Soft ramp-down pattern with net change {magnitude:.3g} across {duration} points ({start_pos}–{end_pos}).",
                "A gentle fall unfolds over {duration} points between {start_pos} and {end_pos}, magnitude {magnitude:.3g}.",
                "Slow de-escalation trend observed with magnitude {magnitude:.3g} over {duration} points ({start_pos}–{end_pos}).",
                "A gradual level reduction (magnitude {magnitude:.3g}) from {start_pos} to {end_pos}.",
                "Longer-span drop event accumulates {magnitude:.3g} across {duration} data points between {start_pos} and {end_pos}.",
                "A sustained downward drift totals {magnitude:.3g} over {duration} points between {start_pos} and {end_pos}."
            ],
            "density_increase": [
                "Density increase anomaly: values clustering together with change magnitude {magnitude:.3g} over {duration} points from position {start_pos} to {end_pos}.",
                "A density increase anomaly is detected, marked by values clustering more closely with magnitude {magnitude:.3g} over {duration} points between positions {start_pos} and {end_pos}.",
                "The time series exhibits a density increase anomaly, with reduced variance and tighter value clustering (magnitude {magnitude:.3g}) over {duration} points from position {start_pos} to {end_pos}.",
                "Values show increased clustering and reduced spread with magnitude {magnitude:.3g} over {duration} points, spanning from position {start_pos} to {end_pos}.",
                "Cluster density rises with reduced dispersion (magnitude {magnitude:.3g}) over {duration} points from {start_pos} to {end_pos}.",
                "A tightening of values indicates higher density (change {magnitude:.3g}) over {duration} points between {start_pos} and {end_pos}.",
                "Variance compresses while values group more closely (magnitude {magnitude:.3g}) across {duration} points from {start_pos} to {end_pos}.",
                "Compactness increases; spread narrows (size {magnitude:.3g}) over {duration} points spanning {start_pos} to {end_pos}.",
                "Observed consolidation of values (magnitude {magnitude:.3g}) over {duration} points from position {start_pos} to {end_pos}.",
                "A denser cluster forms with reduced variability (change {magnitude:.3g}) over {duration} points from {start_pos} to {end_pos}.",
                "Density uptick: tighter grouping with magnitude {magnitude:.3g} over {duration} points between {start_pos} and {end_pos}.",
                "Values converge into a narrower band (magnitude {magnitude:.3g}) over {duration} points from {start_pos} to {end_pos}.",
                "Increased clustering behavior detected (size {magnitude:.3g}) over {duration} points from {start_pos} to {end_pos}."
            ],
            "periodic_pattern_break_by_point": [
                "Periodic pattern break (point): single-point deviation with magnitude {magnitude:.3g} at position {start_pos} (window {duration} point ending at {end_pos}).",
                "A point-wise periodic break occurs, introducing a single outlier of magnitude {magnitude:.3g} within {duration} points from {start_pos} to {end_pos}.",
                "Single-point disruption of periodicity detected (mag {magnitude:.3g}) across {duration} point [{start_pos}, {end_pos}].",
                "A localized periodicity break appears at one point (magnitude {magnitude:.3g}) spanning {duration} sample from {start_pos} to {end_pos}.",
                "Point anomaly interrupts periodic rhythm (size {magnitude:.3g}) in the window {start_pos}–{end_pos} ({duration} point).",
                "A brief point deviation (magnitude {magnitude:.3g}) breaks the periodic pattern across {duration} point from {start_pos} to {end_pos}.",
                "Single-point cadence interruption observed (mag {magnitude:.3g}) over {duration} point between {start_pos} and {end_pos}.",
                "Periodic rhythm shows a lone spike-like deviation (magnitude {magnitude:.3g}) at {start_pos}, windowing {duration} point to {end_pos}.",
                "A solitary departure from cadence (size {magnitude:.3g}) is present within [{start_pos}, {end_pos}] over {duration} point.",
                "An isolated break in periodic order (mag {magnitude:.3g}) is observed at a single index across {duration} point.",
                "One-point anomaly disturbs cycle regularity (magnitude {magnitude:.3g}) in the range {start_pos}–{end_pos} ({duration} point).",
                "Single sample deviates from periodic pattern with magnitude {magnitude:.3g}, affecting {duration} point [{start_pos}, {end_pos}].",
                "A pinpoint irregularity (mag {magnitude:.3g}) disrupts the recurring cycle within {duration} point ({start_pos}–{end_pos}).",
                "Periodic cadence is punctured by a single outlier (magnitude {magnitude:.3g}) across {duration} point.",
                "A one-off deviation from rhythm (size {magnitude:.3g}) emerges at {start_pos} within the {duration}-point window to {end_pos}."
            ],
            "periodic_pattern_break_by_sequence": [
                "Periodic pattern break (sequence): contiguous deviation with magnitude {magnitude:.3g} evolving over {duration} points from {start_pos} to {end_pos}.",
                "A sequence-wise periodic break is detected, showing sustained deviation (mag {magnitude:.3g}) over {duration} consecutive points [{start_pos}, {end_pos}].",
                "Continuous disruption of periodic behavior (magnitude {magnitude:.3g}) persists for {duration} points from {start_pos} to {end_pos}.",
                "Sustained departure from periodic cadence (mag {magnitude:.3g}) across a sequence of {duration} points ({start_pos}–{end_pos}).",
                "A multi-point deviation breaks periodicity (size {magnitude:.3g}) over {duration} samples spanning {start_pos} to {end_pos}.",
                "Run-length periodic break: contiguous drift of {magnitude:.3g} over {duration} points between {start_pos} and {end_pos}.",
                "Sequence anomaly interrupts the periodic rhythm (mag {magnitude:.3g}) for {duration} points from {start_pos} to {end_pos}.",
                "A block of consecutive samples deviates from cycle (magnitude {magnitude:.3g}) across {duration} points [{start_pos}, {end_pos}].",
                "Periodicity falters across a contiguous run (size {magnitude:.3g}) spanning {duration} points.",
                "A streak-level disruption (mag {magnitude:.3g}) undermines the repeating cadence over {duration} points.",
                "Multi-sample drift from periodic baseline (magnitude {magnitude:.3g}) is observed across {duration} points.",
                "An extended sequence departs the rhythmic template (mag {magnitude:.3g}) within window {start_pos}–{end_pos} ({duration} points).",
                "Contiguous deviation accumulates (size {magnitude:.3g}) along {duration} consecutive indices from {start_pos}.",
                "A continuous segment breaks periodic alignment (magnitude {magnitude:.3g}) over {duration} points.",
                "Sustained mismatch against cycle repeats (mag {magnitude:.3g}) across {duration} samples ({start_pos}–{end_pos})."
            ],
            "periodic_pattern_break_by_frequency": [
                "Periodic pattern break (frequency): cycle frequency changes while magnitude remains around {magnitude:.3g} over {duration} points from {start_pos} to {end_pos}.",
                "A frequency anomaly is identified: periodic cadence accelerates or slows with magnitude near {magnitude:.3g} over {duration} points [{start_pos}, {end_pos}].",
                "Frequency shift in periodic pattern (mag ~{magnitude:.3g}) across {duration} points, window {start_pos}–{end_pos}.",
                "Periodic cycle length changes (magnitude ≈{magnitude:.3g}) over {duration} samples from {start_pos} to {end_pos}.",
                "A cadence change alters period (mag ~{magnitude:.3g}) across {duration} points between {start_pos} and {end_pos}.",
                "Period modulation anomaly: frequency deviates while magnitude stays similar ({magnitude:.3g}) for {duration} points ({start_pos}–{end_pos}).",
                "Frequency-modified periodicity observed (magnitude near {magnitude:.3g}) over {duration} points from {start_pos} to {end_pos}.",
                "Cycle rate shifts while peak-to-trough magnitude remains ~{magnitude:.3g} across {duration} points ({start_pos}–{end_pos}).",
                "Tempo of repetition changes (magnitude ≈{magnitude:.3g}) throughout {duration} samples from {start_pos} to {end_pos}.",
                "Period contraction/expansion occurs with similar magnitude (~{magnitude:.3g}) over {duration} points.",
                "Rhythmic spacing varies but wave height stays around {magnitude:.3g} across {duration} points.",
                "Cadence irregularity: frequency multiplier alters cycle length with magnitude near {magnitude:.3g} for {duration} samples.",
                "Phase spacing compresses or dilates while magnitude is stable (~{magnitude:.3g}) over {duration} points.",
                "Frequency anomaly: cycles speed up or slow down, magnitude preserved (~{magnitude:.3g}) within {duration}-point window.",
                "Period shift detected with magnitude consistency (~{magnitude:.3g}) across {duration} points from {start_pos} to {end_pos}."
            ],
            "zero_values": [
                "Zero values anomaly: sustained near-zero values with drop magnitude {magnitude:.3g} over {duration} points from position {start_pos} to {end_pos}.",
                "A zero values anomaly is detected, marked by sustained near-zero readings with magnitude {magnitude:.3g} over {duration} points between positions {start_pos} and {end_pos}.",
                "The time series exhibits a zero values anomaly, with values dropping to or near zero (magnitude {magnitude:.3g}) over {duration} points from position {start_pos} to {end_pos}.",
                "Sustained near-zero value patterns with magnitude {magnitude:.3g} occur over {duration} points, spanning from position {start_pos} to {end_pos}.",
                "Prolonged low-level readings (drop {magnitude:.3g}) persist over {duration} points from {start_pos} to {end_pos}.",
                "Near-zero plateau detected with magnitude {magnitude:.3g} over {duration} points between {start_pos} and {end_pos}.",
                "Values collapse to near zero (size {magnitude:.3g}) and remain for {duration} points spanning {start_pos} to {end_pos}.",
                "Zero-bound behavior appears (magnitude {magnitude:.3g}) over {duration} points from {start_pos} to {end_pos}.",
                "A flat near-zero band persists, drop magnitude {magnitude:.3g}, over {duration} points between {start_pos} and {end_pos}.",
                "Sustained suppression towards zero (mag {magnitude:.3g}) over {duration} points from {start_pos} to {end_pos}.",
                "Zero-value interval spans {duration} points with magnitude {magnitude:.3g} from {start_pos} to {end_pos}.",
                "Extended near-zero segment observed (size {magnitude:.3g}) over {duration} points between {start_pos} and {end_pos}.",
                "Persistent zero-ish readings indicate a drop of {magnitude:.3g} over {duration} points from {start_pos} to {end_pos}."
            ],
            "saturation_values": [
                "Saturation anomaly: values reaching upper limits with rise magnitude {magnitude:.3g} over {duration} points from position {start_pos} to {end_pos}.",
                "A saturation anomaly is identified, characterized by values reaching their upper bounds with magnitude {magnitude:.3g} over {duration} points between positions {start_pos} and {end_pos}.",
                "The time series shows a saturation anomaly, with values hitting maximum limits (magnitude {magnitude:.3g}) over {duration} points from position {start_pos} to {end_pos}.",
                "Values reach saturation levels and remain at upper limits with magnitude {magnitude:.3g} over {duration} points, spanning from position {start_pos} to {end_pos}.",
                "Prolonged upper-bound readings (rise {magnitude:.3g}) persist over {duration} points from {start_pos} to {end_pos}.",
                "Ceiling-level plateau detected with magnitude {magnitude:.3g} over {duration} points between {start_pos} and {end_pos}.",
                "Values climb to saturation (size {magnitude:.3g}) and hold for {duration} points spanning {start_pos} to {end_pos}.",
                "Upper-limit behavior appears (magnitude {magnitude:.3g}) over {duration} points from {start_pos} to {end_pos}.",
                "A flat max-level band persists, rise magnitude {magnitude:.3g}, over {duration} points between {start_pos} and {end_pos}.",
                "Sustained elevation towards maximum (mag {magnitude:.3g}) over {duration} points from {start_pos} to {end_pos}.",
                "Saturation interval spans {duration} points with magnitude {magnitude:.3g} from {start_pos} to {end_pos}.",
                "Extended near-ceiling segment observed (size {magnitude:.3g}) over {duration} points between {start_pos} and {end_pos}.",
                "Persistent upper-limit readings indicate a rise of {magnitude:.3g} over {duration} points from {start_pos} to {end_pos}."
            ],
            "volatility_increase": [
                "Volatility increase anomaly: fluctuation amplitude expands with change magnitude {magnitude:.3g} over {duration} points from position {start_pos} to {end_pos}.",
                "A volatility increase anomaly is detected, marked by a clear rise in variability (magnitude {magnitude:.3g}) over {duration} points between positions {start_pos} and {end_pos}.",
                "The time series exhibits a volatility increase anomaly, with wider swings and stronger jitter (magnitude {magnitude:.3g}) over {duration} points from position {start_pos} to {end_pos}.",
                "Variability increases noticeably with magnitude {magnitude:.3g} across {duration} points, spanning from position {start_pos} to {end_pos}.",
                "A high-variance regime emerges (size {magnitude:.3g}) over {duration} points from {start_pos} to {end_pos}.",
                "Fluctuation spread broadens with magnitude {magnitude:.3g} for {duration} points between {start_pos} and {end_pos}.",
                "Oscillation intensity rises (mag {magnitude:.3g}) across {duration} samples from {start_pos} to {end_pos}.",
                "The signal enters a noisier, more variable band with change {magnitude:.3g} over {duration} points in window {start_pos}-{end_pos}.",
                "A sustained increase in variance is observed (magnitude {magnitude:.3g}) over {duration} points from {start_pos} to {end_pos}.",
                "Volatility uplift detected: larger peak-to-trough dispersion (magnitude {magnitude:.3g}) over {duration} points between {start_pos} and {end_pos}.",
                "The amplitude envelope expands with magnitude {magnitude:.3g}, lasting {duration} points from {start_pos} to {end_pos}.",
                "An elevated-variability interval persists for {duration} points with change size {magnitude:.3g} across {start_pos} to {end_pos}.",
                "Volatility spikes to a higher level (mag {magnitude:.3g}) throughout {duration} points in range [{start_pos}, {end_pos}]."
            ],
            "volatility_decrease": [
                "Volatility decrease anomaly: fluctuation amplitude contracts with change magnitude {magnitude:.3g} over {duration} points from position {start_pos} to {end_pos}.",
                "A volatility decrease anomaly is detected, marked by a clear drop in variability (magnitude {magnitude:.3g}) over {duration} points between positions {start_pos} and {end_pos}.",
                "The time series exhibits a volatility decrease anomaly, with narrower swings and reduced jitter (magnitude {magnitude:.3g}) over {duration} points from position {start_pos} to {end_pos}.",
                "Variability declines noticeably with magnitude {magnitude:.3g} across {duration} points, spanning from position {start_pos} to {end_pos}.",
                "A low-variance regime emerges (size {magnitude:.3g}) over {duration} points from {start_pos} to {end_pos}.",
                "Fluctuation spread narrows with magnitude {magnitude:.3g} for {duration} points between {start_pos} and {end_pos}.",
                "Oscillation intensity falls (mag {magnitude:.3g}) across {duration} samples from {start_pos} to {end_pos}.",
                "The signal enters a calmer, less variable band with change {magnitude:.3g} over {duration} points in window {start_pos}-{end_pos}.",
                "A sustained decrease in variance is observed (magnitude {magnitude:.3g}) over {duration} points from {start_pos} to {end_pos}.",
                "Volatility reduction detected: smaller peak-to-trough dispersion (magnitude {magnitude:.3g}) over {duration} points between {start_pos} and {end_pos}.",
                "The amplitude envelope compresses with magnitude {magnitude:.3g}, lasting {duration} points from {start_pos} to {end_pos}.",
                "A reduced-variability interval persists for {duration} points with change size {magnitude:.3g} across {start_pos} to {end_pos}.",
                "Volatility drops to a lower level (mag {magnitude:.3g}) throughout {duration} points in range [{start_pos}, {end_pos}]."
            ],
            "historical_pattern_mismatch": [
                "Historical pattern mismatch: values differ from historical patterns with deviation magnitude {magnitude:.3g} over {duration} points from position {start_pos} to {end_pos}.",
                "A historical pattern mismatch anomaly is detected, marked by deviation from historical norms with magnitude {magnitude:.3g} over {duration} points between positions {start_pos} and {end_pos}.",
                "The time series exhibits a historical pattern mismatch anomaly, with values significantly different from historical patterns (magnitude {magnitude:.3g}) over {duration} points from position {start_pos} to {end_pos}.",
                "Values deviate significantly from historical patterns with magnitude {magnitude:.3g} over {duration} points, spanning from position {start_pos} to {end_pos}.",
                "Departure from historical baseline detected (deviation {magnitude:.3g}) over {duration} points from {start_pos} to {end_pos}.",
                "A mismatch against historical profile appears with size {magnitude:.3g} over {duration} points between {start_pos} and {end_pos}.",
                "Values diverge from learned historical behavior (mag {magnitude:.3g}) over {duration} points spanning {start_pos} to {end_pos}.",
                "Historical conformity breaks, deviation magnitude {magnitude:.3g} over {duration} points from {start_pos} to {end_pos}.",
                "Patterns shift away from historical norms (size {magnitude:.3g}) over {duration} points between {start_pos} and {end_pos}.",
                "A historical drift/mismatch (mag {magnitude:.3g}) persists over {duration} points from {start_pos} to {end_pos}.",
                "Mismatch relative to historical template detected, deviation {magnitude:.3g} over {duration} points spanning {start_pos} to {end_pos}.",
                "Historical pattern violation observed (magnitude {magnitude:.3g}) over {duration} points from {start_pos} to {end_pos}.",
                "Significant deviation from historical expectation ({magnitude:.3g}) over {duration} points between {start_pos} and {end_pos}."
            ]
        }

        self.common_desc_pool = {
            "no-anomaly": [
                "No obvious anomalies detected.",
                "The time series appears normal without significant anomalies.",
                "No notable anomalies are present in the time series.",
                "The time series shows no signs of anomalies.",
                "Overall, the time series appears stable and consistent without anomalies.",
                "No significant anomalies are observed in the time series.",
                "The time series maintains normal behavior without any anomalies.",
                "No abnormal patterns or anomalies are detected in the time series.",
                "No irregular behaviors are found; the series remains within expected bounds.",
                "The signal is steady with no deviations indicative of anomalies.",
                "Observations remain consistent; nothing anomalous stands out.",
                "No departures from typical patterns are identified in this interval.",
                "The series aligns with historical behavior; no anomaly cues are present.",
                "Measurements stay within normal variability, showing no anomalies.",
                "No out-of-pattern events are observed throughout the examined range.",
                "The profile is stable and conforms to expected dynamics without anomalies.",
                "No anomaly triggers are activated; the data looks routine.",
                "Everything appears normal; no suspicious segments detected.",
                "The time series maintains expected seasonality/trend with no abnormal signals.",
                "No unusual fluctuations or structural breaks are evident."
            ],
            "first-inspect": [
                "Initially, we will conduct anomaly detection followed by slicing a wider window around each detected anomaly for detailed analysis.",
                "Our first step will be to detect anomalies and then extract a larger window around each anomaly for closer examination.",
                "We plan to start with anomaly detection and subsequently slice a broader window for each anomaly to facilitate in-depth inspection.",
                "First, we run detection, then expand the inspection window around candidates for clarity.",
                "We begin by identifying anomaly candidates and widen the surrounding context for review.",
                "Step one: detect anomalies; step two: slice broader windows per anomaly for verification.",
                "The initial pass performs detection, followed by context expansion around each flagged segment.",
                "Detection comes first; we then create extended slices to examine local behavior.",
                "Start with an anomaly scan, then extract wider windows to validate each finding.",
                "We kick off with detection and pad each anomaly with a larger window for detailed checks.",
                "Run detection, then open up the time window around each detection for fine-grained assessment.",
                "Phase 1 identifies anomalies; phase 2 produces wider slices for human-readable inspection.",
                "Initial inspection consists of detection plus a generous windowing around identified anomalies.",
                "We first detect anomalies and then expand the analysis window to capture surrounding context."
            ],
            "further-inspect": [
                "We further analyze across unconfirmed anomalies for {step_str} detection based on the previous slices.",
                "Further inspection is conducted on unconfirmed anomalies for {step_str} detection using the previously sliced windows.",
                "We proceed with additional analysis on unconfirmed anomalies for {step_str} detection, leveraging the earlier extracted slices.",
                "Additional inspection is performed on unconfirmed anomalies for {step_str} detection, utilizing the previously obtained slices.",
                "We re-evaluate unconfirmed segments for {step_str} detection using the prior context windows.",
                "A secondary pass targets unconfirmed anomalies for {step_str} detection, anchored on earlier slices.",
                "We perform follow-up checks on pending anomalies for {step_str} detection with the existing windows.",
                "Further scrutiny focuses on unresolved cases for {step_str} detection, reusing previous slices.",
                "We iterate over unconfirmed detections for {step_str} detection leveraging the already extracted ranges.",
                "Subsequent analysis revisits unconfirmed anomalies for {step_str} detection with contextual windows.",
                "We apply additional verification on tentative anomalies for {step_str} detection based on prior slicing.",
                "Another round of inspection is run on uncertain anomalies for {step_str} detection, using the saved windows.",
                "We extend analysis to unresolved anomalies for {step_str} detection while maintaining windowed context.",
                "We refine judgments on unconfirmed anomalies for {step_str} detection by reusing earlier slices.",
                "We conduct targeted re-checks on unconfirmed anomalies for {step_str} detection with the same windowing."
            ],
            "slice": [
                "For anomaly {idx_i}, slicing window [{st}, {en}] to refine.",
                "Refining anomaly {idx_i} by slicing window [{st}, {en}].",
                "Slicing window [{st}, {en}] for detailed analysis of anomaly {idx_i}.",
                "Extracting window [{st}, {en}] to focus on anomaly {idx_i}.",
                "Zooming into window [{st}, {en}] to better understand anomaly {idx_i}.",
                "We narrow the view to [{st}, {en}] to scrutinize anomaly {idx_i}.",
                "Creating a focused slice [{st}, {en}] for anomaly {idx_i} inspection.",
                "Window [{st}, {en}] is selected to refine the evaluation of anomaly {idx_i}.",
                "We isolate range [{st}, {en}] to analyze anomaly {idx_i} more precisely.",
                "A detailed window [{st}, {en}] is carved out for anomaly {idx_i}.",
                "Setting up slice [{st}, {en}] to validate anomaly {idx_i} behavior.",
                "Examining anomaly {idx_i} within the slice [{st}, {en}] for clarity.",
                "We extract context [{st}, {en}] to assess anomaly {idx_i}.",
                "Focusing analysis on [{st}, {en}] to verify anomaly {idx_i} characteristics."
            ],
            "anomaly-with-idx": [
                "Anomaly {idx_i} with {pat} pattern: {desc}",
                "We inspect the slice for anomaly {idx_i} with {pat} pattern: {desc}",
                "Detected anomaly {idx_i} exhibiting {pat} pattern: {desc}",
                "Anomaly {idx_i} characterized by {pat} pattern: {desc}",
                "The slice reveals anomaly {idx_i} with {pat} pattern: {desc}",
                "Examining anomaly {idx_i}, pattern {pat} observed: {desc}",
                "Anomaly {idx_i} presents a {pat} signature: {desc}",
                "Slice analysis indicates anomaly {idx_i} of type {pat}: {desc}",
                "We see anomaly {idx_i} following {pat} characteristics: {desc}",
                "The window highlights anomaly {idx_i} ({pat}): {desc}",
                "Anomaly {idx_i} appears consistent with {pat}: {desc}",
                "Observed anomaly {idx_i} conforms to {pat} pattern: {desc}",
                "Review shows anomaly {idx_i} in {pat} form: {desc}",
                "Anomaly {idx_i} exhibits traits of {pat}: {desc}"
            ],
            "get-raw-idx": [
                "Considering the relative offset of {relative_offset}, the interval in original series is ({st}, {en}).",
                "Adjusting for a relative offset of {relative_offset}, the original series interval is ({st}, {en}).",
                "With a relative offset of {relative_offset}, the corresponding interval in the original series is ({st}, {en}).",
                "Taking into account a relative offset of {relative_offset}, the original series interval becomes ({st}, {en}).",
                "Factoring in a relative offset of {relative_offset}, the interval in the original series is ({st}, {en}).",
                "After applying offset {relative_offset}, the raw series interval maps to ({st}, {en}).",
                "Offset {relative_offset} yields the raw indices ({st}, {en}) in the original sequence.",
                "Considering offset {relative_offset}, we recover original bounds ({st}, {en}).",
                "Given {relative_offset}, the interval translates back to raw indices ({st}, {en}).",
                "Accounting for {relative_offset}, the original window is ({st}, {en}).",
                "Applying the relative shift {relative_offset}, we obtain raw interval ({st}, {en}).",
                "With shift {relative_offset}, the corresponding raw window is ({st}, {en}).",
                "Using {relative_offset} correction, original indices are ({st}, {en}).",
                "After correction by {relative_offset}, the raw segment is ({st}, {en})."
            ],
            "consistent": [
                "The result is consistent with the last detection; we confirm and stop for this anomaly.",
                "The outcome aligns with the previous detection; we confirm and conclude for this anomaly.",
                "The findings are in agreement with the last detection; we confirm and finalize for this anomaly.",
                "The result matches the prior detection; we confirm and end the process for this anomaly.",
                "The detection outcome is consistent with the last analysis; we confirm and cease further action for this anomaly.",
                "Consistency with the prior result is observed; we confirm and close this case.",
                "The re-check agrees with the previous decision; confirm and terminate handling for this anomaly.",
                "Results corroborate earlier detection; confirmation issued and processing stops.",
                "Outcome remains unchanged; we confirm and mark this anomaly as finalized.",
                "The verification aligns with the last pass; we confirm and wrap up.",
                "Findings are stable relative to the last run; confirm and conclude.",
                "No discrepancy versus the prior detection; we confirm and finish.",
                "Agreement achieved with previous analysis; confirm and end.",
                "The anomaly status is confirmed consistent; no further action required."
            ],
            "inconsistent-then-slice": [
                "It's not consistent with the last detection; we will slice [{st}, {en}] for further checking.",
                "The result differs from the previous detection; we plan to extract window [{st}, {en}] for additional examination.",
                "The outcome is inconsistent with the last detection; we will focus on window [{st}, {en}] for further analysis.",
                "The findings do not align with the prior detection; we will slice [{st}, {en}] to investigate further.",
                "The detection result contradicts the last analysis; we will extract window [{st}, {en}] for deeper inspection.",
                "Inconsistency observed; proceed to slice [{st}, {en}] for validation.",
                "We notice divergence from the prior result; create window [{st}, {en}] to re-check.",
                "Mismatch detected; isolate [{st}, {en}] to clarify the anomaly status.",
                "Since results disagree, we expand slice [{st}, {en}] for further evidence.",
                "The current outcome conflicts with the previous; inspect via window [{st}, {en}].",
                "Disagreement found; narrow to [{st}, {en}] for closer inspection.",
                "Not aligned with earlier detection; examine context within [{st}, {en}].",
                "We will re-slice to [{st}, {en}] to reconcile the discrepancy.",
                "Incongruent signals present; analyze the segment [{st}, {en}] in detail."
            ],
            "inconsistent-then-confirm": [
                "It's not consistent with the last detection; however, the maximum number of slicing is reached. Anomaly {idx_i} is considered as false positive and will be ignored.",
                "The result differs from the previous detection; nevertheless, we've reached the slicing limit. We regard anomaly {idx_i} as a false positive and will disregard it.",
                "The outcome is inconsistent with the last detection; but since the maximum slicing attempts are exhausted, anomaly {idx_i} will be treated as a false positive and ignored.",
                "The findings do not align with the prior detection; however, with the slicing limit reached, we consider anomaly {idx_i} a false positive and will overlook it.",
                "The detection result contradicts the last analysis; yet, having hit the slicing maximum, anomaly {idx_i} is deemed a false positive and will be dismissed.",
                "Despite inconsistency, slicing quota is exhausted; mark anomaly {idx_i} as false positive and ignore.",
                "Slicing limit reached; anomaly {idx_i} remains inconsistent and is discarded as false positive.",
                "Further slicing is not permitted; treat anomaly {idx_i} as a false positive.",
                "We cannot slice further; anomaly {idx_i} is concluded to be a false positive.",
                "Resolution by limit: anomaly {idx_i} is inconsistent and will be ignored.",
                "Maximum refinement attempts used; dismiss anomaly {idx_i} as false positive.",
                "Cap reached on slicing; anomaly {idx_i} is considered non-actionable and ignored.",
                "No more slices allowed; anomaly {idx_i} classified as false positive.",
                "Slicing threshold hit; we finalize anomaly {idx_i} as a false positive."
            ],
            "check-anomaly-with-idx-and-history": [
                "For anomaly {idx_i} ({pat}), we combine same-pattern history, other anomalies in the current window, and the human intent to make the final decision.",
                "Analyzing anomaly {idx_i} ({pat}) by jointly considering matched historical anomalies, neighboring anomalies in the current window, and human intent.",
                "For anomaly {idx_i} ({pat}), the judgment is based on three signals: history of the same pattern, other anomalies in the current window, and human intent constraints.",
                "We reassess anomaly {idx_i} ({pat}) using same-pattern memory evidence, context from other anomalies in this window, and the human intent rules.",
                "Decision for anomaly {idx_i} ({pat}) is made by integrating historical same-pattern events, current-window anomaly context, and human intent.",
                "For anomaly {idx_i} ({pat}), we synthesize retrieved same-pattern history with current-window anomaly interactions under the human intent.",
                "We evaluate anomaly {idx_i} ({pat}) with a combined view of pattern-matched history, other anomalies in the current window, and human intent guidance.",
                "The analysis of anomaly {idx_i} ({pat}) merges memory hits for the same pattern, concurrent window anomalies, and human intent constraints.",
                "For anomaly {idx_i} ({pat}), final filtering relies on historical pattern recurrence, current-window anomaly context, and human intent.",
                "Anomaly {idx_i} ({pat}) is assessed through joint reasoning over same-pattern history, other anomalies in the current window, and human intent."
            ],
            "check-anomaly-with-idx": [
                "For anomaly {idx_i}, we consider the human intent to make the final decision.",
                "Analyzing anomaly {idx_i} with respect to the human intent for final judgment.",
                "For anomaly {idx_i}, the decision is based on the human intent constraints.",
                "We reassess anomaly {idx_i} under the guidance of the human intent rules.",
                "Decision for anomaly {idx_i} is made by integrating the human intent.",
                "For anomaly {idx_i}, we synthesize the human intent rules to guide the final decision.",
                "We evaluate anomaly {idx_i} with a combined view of the human intent guidance.",
                "The analysis of anomaly {idx_i} merges the human intent constraints.",
                "For anomaly {idx_i}, final filtering relies on the human intent.",
                "Anomaly {idx_i} is assessed through reasoning over the human intent."
            ],

        }

        # Diversified description pool for human intent rules
        self.intent_rule_desc_pool = {
            # Focus rules
            "focus_on_increase_only": [
                "Only consider increase-type anomalies; non-increase patterns are excluded.",
                "Scope limited to upward anomalies; decreases and non-increase forms are ignored.",
                "Focus strictly on increase-side signals; other types are out of scope.",
                "Retain only anomalies showing an increase; exclude decrease/zero patterns.",
                "Report increases only; drop any decrease or neutral patterns.",
                "Keep upward deviations; discard downward and flat anomalies.",
                "Restrict to increase-direction events; exclude opposite-direction cases.",
                "Restrict to positive-direction anomalies; others are not considered.",
                "Select increase-oriented signals; suppress all non-increase types.",
                "Confine attention to rising anomalies; ignore falling or zero-bound ones."
            ],
            "focus_on_decrease_only": [
                "Only consider decrease-type anomalies; non-decrease patterns are excluded.",
                "Scope limited to downward anomalies; increases are ignored.",
                "Focus strictly on decrease-side signals; other types are out of scope.",
                "Retain only anomalies showing a decrease; exclude increase-oriented patterns.",
                "Report decreases only; drop any increase or neutral patterns.",
                "Keep downward deviations; discard upward and flat anomalies.",
                "Restrict to decrease-direction events; exclude opposite-direction cases.",
                "Restrict to negative-direction anomalies; others are not considered.",
                "Select decrease-oriented signals; suppress all non-decrease types.",
                "Confine attention to falling anomalies; ignore rising or zero-bound ones."
            ],
            "focus_on_zero_only": [
                "Only report sustained zero/near-zero anomalies; all other patterns are ignored.",
                "Restrict analysis to zero-values patterns exclusively.",
                "Focus solely on zero-bound behavior; non-zero anomalies are excluded.",
                "Keep only zero-values anomalies; discard other types.",
                "Only zero/near-zero segments are in scope; other anomalies are excluded.",
                "Track zero-bound anomalies exclusively; ignore non-zero events.",
                "Restrict to outages/zero patterns; non-zero behaviors are excluded.",
                "Zero-centric focus: keep near-zero anomalies only.",
                "Concentrate on zero-value anomalies; drop all other patterns.",
                "Zero-only mode: exclude every non-zero pattern."
            ],

            # Threshold rule
            "absolute_magnitude_threshold": [
                "Enforce a minimum magnitude of {threshold_value_str}; smaller anomalies are removed.",
                "Only keep anomalies with magnitude at least {threshold_value_str}.",
                "Exclude anomalies whose magnitude is below {threshold_value_str}.",
                "Require magnitude at least {threshold_value_str} to be retained.",
                "Discard anomalies with magnitude under {threshold_value_str}.",
                "Minimum required magnitude: {threshold_value_str}; smaller ones are excluded.",
                "Keep items reaching at least {threshold_value_str} in magnitude.",
                "Apply an magnitude floor of {threshold_value_str} to exclude minor noise.",
                "Thresholding by magnitude at {threshold_value_str}; sub-threshold events are removed.",
                "Accept anomalies only if magnitude is {threshold_value_str} or higher."
            ],

            # Frequency rule
            "ignore_frequent_patterns": [
                "Ignore patterns occurring at least {min_occurrences} times within any window of {window_size} points.",
                "Treat patterns as noise if they repeat at least {min_occurrences} times in a sliding window of {window_size} points.",
                "Frequent or periodic patterns that appear at least {min_occurrences} times in a window of {window_size} points will be ignored.",
                "Exclude patterns that repeat at least {min_occurrences} times inside any span of {window_size} points.",
                "Ignore patterns that repeat at least {min_occurrences} times within a window of {window_size} points.",
                "Exclude periodic or frequent patterns seen at least {min_occurrences} times in every {window_size} points.",
                "Treat as background any pattern that recurs at least {min_occurrences} times in a span of {window_size} points.",
                "De-emphasize patterns with occurrence counts of at least {min_occurrences} within a {window_size}-point window.",
                "Suppress motifs that appear at least {min_occurrences} times across a sliding window of {window_size} points.",
                "Classify patterns as frequent and ignore them when occurrences reach at least {min_occurrences} within a {window_size}-point window."
            ],
            "ignore_rare_patterns": [
                "Ignore occasional patterns that appear at most {max_occurrences} times within any window of {window_size} points.",
                "Treat low-frequency patterns as ignorable if they appear at most {max_occurrences} times in a sliding window of {window_size} points.",
                "Exclude patterns considered too sporadic, with no more than {max_occurrences} occurrences per {window_size}-point window.",
                "Ignore rare patterns whose count stays within {max_occurrences} inside a {window_size}-point span.",
                "Suppress occasional motifs that occur no more than {max_occurrences} times over {window_size} points.",
                "Drop infrequent patterns when their repetition count is no higher than {max_occurrences} in each {window_size}-point window.",
                "De-emphasize sparse patterns that appear at most {max_occurrences} times in a window of {window_size} points.",
                "Filter out low-occurrence patterns that stay at or below {max_occurrences} within a {window_size}-point sliding window.",
                "Ignore one-off or rare repeats when occurrence counts stay at or below {max_occurrences} across {window_size} points.",
                "Exclude anomalies from patterns deemed occasional under this rule, meaning no more than {max_occurrences} occurrences in {window_size} points.",
            ],

            # Ignore short-duration anomalies (human_intent_generator)
            "ignore_short_duration": [
                "Ignore anomalies with duration < {min_duration} points.",
                "Exclude short-lived anomalies whose duration is under {min_duration} points.",
                "Drop anomalies shorter than {min_duration} points in duration.",
                "Keep only anomalies lasting at least {min_duration} points.",
                "Apply a minimum duration of {min_duration} points; shorter cases are excluded.",
                "Suppress brief anomalies (< {min_duration} points).",
                "Only keep anomalies that persist for {min_duration} points or longer.",
                "Treat anomalies shorter than {min_duration} points as non-actionable and exclude them.",
                "Enforce a duration floor of {min_duration} points; anything shorter is removed.",
                "Filter out transient spikes shorter than {min_duration} points and keep sustained ones.",
            ],

            # Truncation rule
            "truncate_overlong_duration": [
                "If an anomaly lasts over {max_duration} points, keep only the first {max_duration} points.",
                "Truncate overly long anomalies to {max_duration} points.",
                "Cap anomaly duration at {max_duration} points; discard the rest.",
                "Limit each anomaly to {max_duration} points when it exceeds that length.",
                "Cut anomalies longer than {max_duration} points to {max_duration} points.",
                "Trim any overlength anomaly down to {max_duration} points.",
                "Any duration exceeding {max_duration} points is truncated to this cap.",
                "Apply hard cap of {max_duration} points to anomaly duration.",
                "Limit anomaly length to {max_duration} points; excess is dropped.",
                "Clamp anomaly duration at {max_duration} points if it runs longer."
            ],

            # Merge close anomalies (human_intent_generator)
            "merge_close_anomalies": [
                "Merge anomalies that are close (gap <= {max_gap}).",
                "Merge neighboring anomalies when the gap is at most {max_gap} points.",
                "Combine anomalies that occur close together (gap <= {max_gap}).",
                "Merge close events within {max_gap} points to form a single anomaly.",
                "If anomalies are separated by <= {max_gap} points, merge them.",
                "Group nearby anomalies (gap <= {max_gap}) into one consolidated event.",
                "Collapse close-by anomalies into a single segment when the separation is <= {max_gap}.",
                "Fuse anomalies with small gaps (<= {max_gap}) to simplify reporting.",
                "When anomalies are almost contiguous (gap <= {max_gap}), report them as one merged anomaly.",
                "Merge adjacent/near-adjacent anomalies using a maximum gap of {max_gap} points.",
            ],

            # Others intents (human_intent_generator)
            "limit_total_reports": [
                "Keep at most {max_reports} anomaly reports, prioritizing {rank_by_nl}.",
                "Limit the final output to {max_reports} anomalies, giving priority to {rank_by_nl}.",
                "Return no more than {max_reports} anomalies and rank them by {rank_by_nl}.",
                "Cap anomaly reports at {max_reports}, retaining cases with {rank_by_nl} first.",
                "Only keep up to {max_reports} anomalies, selected by {rank_by_nl}.",
                "Restrict the final anomaly list to {max_reports} entries, ordered by {rank_by_nl}.",
                "Apply a reporting budget of {max_reports} and prioritize anomalies with {rank_by_nl}.",
                "Trim final detections to {max_reports}, preferring anomalies with {rank_by_nl}.",
                "Set a maximum of {max_reports} returned anomalies, ranked by {rank_by_nl}.",
                "Enforce a {max_reports}-item reporting cap while prioritizing {rank_by_nl}.",
            ],

            # Scope intents (human_intent_generator): expand what to detect (not used for exclusion)
            "include_slow_trends": [
                "Also detect slow trend change anomalies.",
                "Include slow trend changes in the anomaly types to report.",
                "Expand scope to cover slow increases/decreases.",
                "Add slow increase/decrease patterns to the detection scope.",
                "In addition to abrupt changes, look for gradual trend shifts.",
                "Enable detection of slow trend drift anomalies.",
                "Consider slow-changing trend anomalies as in-scope.",
                "Broaden detection to include gradual upward/downward drifts.",
                "Please also flag slow trend change events.",
                "Extend coverage to slow trend anomalies beyond spikes/dips.",
            ],
            "include_small_baseline_shifts": [
                "Also detect subtle but sustained baseline shifts (min_duration={min_duration}, min_offset_ratio={min_offset_ratio}).",
                "Expand scope to include small baseline shifts with min_duration={min_duration} and min_offset_ratio={min_offset_ratio}.",
                "Include subtle baseline shifts (min_duration={min_duration}, min_offset_ratio={min_offset_ratio}).",
                "Add subtle baseline shift anomalies into scope (min_duration={min_duration}, min_offset_ratio={min_offset_ratio}).",
                "Detect small but persistent baseline moves with min_offset_ratio={min_offset_ratio} over at least {min_duration} points.",
                "Also consider minor baseline regime shifts that last >= {min_duration} points.",
                "Include sustained baseline offsets (min_duration={min_duration}, min_offset_ratio={min_offset_ratio}).",
                "Broaden scope to capture small baseline shifts that persist for {min_duration} points.",
                "Please detect subtle baseline changes with offset ratio >= {min_offset_ratio}.",
                "Enable detection of gentle baseline shift anomalies under the given duration/ratio constraints.",
            ],
            "include_volatility_changes": [
                "Also detect volatility changes when the volatility multiplier is no smaller than {variance_ratio}.",
                "Expand scope to include volatility shifts where the volatility multiplier is at least {variance_ratio}.",
                "Include volatility increase/decrease anomalies with a volatility-change factor of at least {variance_ratio}.",
                "Add volatility regime changes to the scope when the variability ratio is no smaller than {variance_ratio}.",
                "Also flag sustained volatility increases/decreases when the variability ratio is at least {variance_ratio}.",
                "Detect changes in variability when the fluctuation scale is no smaller than {variance_ratio} times baseline.",
                "Include anomalies where volatility shifts reach at least {variance_ratio}x relative to baseline variability.",
                "Broaden detection to cover volatility changes with a minimum volatility factor of {variance_ratio}.",
                "Please report volatility change events with a variability ratio no smaller than {variance_ratio}, not just mean/level shifts.",
                "Enable volatility shift detection using a minimum fluctuation ratio of {variance_ratio}.",
            ],

            # Write text pool for intent:
            # "ignore_recent_reoccurrence": {
			# 	"probability": 0.30,
			# 	"category": "others",
			# 	"anomalies": anomalies,
			# 	"parameter_ranges": {
			# 		"window_size": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
			# 		"mode": ["magnitude", "duration", "both"],
			# 	"description_template": "Ignore recent reoccurrence in a window of {window_size} points by {mode} comparison",
            "ignore_recent_reoccurrence": [
                "Within the last {window_size} points, ignore a recurring similar anomaly if its {ignore_recent_mode_or_nl} does not exceed historical ones.",
                "Use a {window_size}-point look-back window: if a similar anomaly reoccurs and its {ignore_recent_mode_or_nl} is not higher than past cases, ignore it.",
                "In a recent window of {window_size} points, recurring similar anomalies are filtered when {ignore_recent_mode_or_nl} does not surpass historical anomalies.",
                "Apply recent-reoccurrence filtering over {window_size} points; ignore similar repeats whose {ignore_recent_mode_or_nl} is not greater than previous ones.",
                "For similar anomalies seen in the past {window_size} points, keep only those with {ignore_recent_mode_or_nl} higher than historical records.",
                "If a similar pattern appears again within {window_size} points and its {ignore_recent_mode_or_nl} is not larger than before, treat it as ignorable.",
                "Using a {window_size}-point memory window, suppress repeated similar anomalies unless their {ignore_recent_mode_or_nl} exceeds prior anomalies.",
                "Ignore near-term repeats in the last {window_size} points when {ignore_recent_mode_or_nl} fails to beat historical counterparts.",
                "A recurring similar anomaly in the recent {window_size}-point range is ignored if {ignore_recent_mode_or_nl} does not outperform previous events.",
                "For recent repeats within {window_size} points, the anomaly is dropped when {ignore_recent_mode_or_nl} is not above historical same-pattern anomalies."
            ],

            # Fallback generic wording
            "_generic": [
                "Apply this intent criterion to refine anomalies.",
                "Use this rule to prune less relevant anomalies.",
                "This rule narrows down anomalies according to its constraints.",
                "Refine detections by enforcing this intent condition.",
                "Use this intent constraint to improve precision by removing noise.",
                "Tighten results by applying this rule's constraints.",
                "Refine the candidate set according to this rule.",
                "Reduce false positives via this intent criterion.",
                "Constrain anomalies using this rule to focus on relevance.",
                "Apply rule-based pruning to streamline detections."
            ]
        }

        # Diversified reason pool per rule (10 templates each)
        self.intent_reason_pool = {
            "focus_on_increase_only": [
                "the pattern is not an increase pattern",
                "the pattern belongs outside the increase-only scope",
                "the signal is not upward and falls outside the increase restriction",
                "the increase-only focus does not admit this pattern",
                "the pattern does not align with the increase-only selection",
                "only upward deviations are kept and this pattern is not upward",
                "the policy retains increases and this pattern is non-increase",
                "the increase-centric intent does not keep this pattern",
                "the pattern conflicts with the increase-focused rule",
                "the scope is limited to increases and this pattern is removed"
            ],
            "focus_on_decrease_only": [
                "the pattern is not a decrease pattern",
                "the pattern belongs outside the decrease-only scope",
                "the signal is not downward and falls outside the decrease restriction",
                "the decrease-only focus does not admit this pattern",
                "the pattern does not align with the decrease-only selection",
                "only downward deviations are kept and this pattern is not downward",
                "the policy retains decreases and this pattern is non-decrease",
                "the decrease-centric intent does not keep this pattern",
                "the pattern conflicts with the decrease-focused rule",
                "the scope is limited to decreases and this pattern is removed"
            ],
            "focus_on_zero_only": [
                "the pattern is not a zero or near-zero pattern",
                "the rule keeps zero or near-zero values only and this pattern is not zero-bound",
                "the pattern lies outside the zero-value scope",
                "only zero-bound anomalies are retained and this pattern is not zero-bound",
                "the zero-centric intent does not admit this pattern",
                "the pattern does not match the zero-only criteria",
                "the pattern is non-zero and is excluded under the zero-only rule",
                "the zero-exclusive intent excludes this pattern",
                "the pattern is not zero-bound and is excluded",
                "we retain zero values only and this pattern is removed"
            ],
            "absolute_magnitude_threshold": [
                "the magnitude {mag} is below the threshold {thr}",
                "the value fails the minimum magnitude {thr} with an magnitude of {mag}",
                "the magnitude {mag} is under the required threshold {thr}",
                "the required magnitude of at least {thr} is not met because the value is {mag}",
                "the magnitude floor {thr} excludes it since the value is {mag}",
                "the magnitude {mag} is too small compared with the required {thr}",
                "the magnitude requirement {thr} is not satisfied by the value {mag}",
                "the magnitude is below the gate of {thr} with a value of {mag}",
                "the magnitude falls short of the required {thr} with {mag}",
                "the threshold {thr} is unmet because the magnitude is {mag}"
            ],
            "ignore_frequent_patterns": [
                "the pattern repeats frequently within the examined windows",
                "the pattern is classified as frequent based on repetition",
                "the pattern recurs too often across the series",
                "the frequency exceeds tolerance with at least {min_occ} occurrences per {win} points",
                "the occurrence density is high and marks the pattern as frequent",
                "the behavior is treated as background repetition with at least {min_occ} occurrences per {win} points",
                "the frequent motif threshold is met with at least {min_occ} occurrences in {win} points",
                "the repetition rate is too high for retention",
                "the repeating pattern exceeds the limit with at least {min_occ} occurrences in {win} points",
                "the pattern frequency crosses the configured limit"
            ],
            "ignore_rare_patterns": [
                "the pattern appears only occasionally within the examined windows",
                "the pattern is classified as low-frequency based on repetition",
                "the pattern recurs too sparsely across the series",
                "the occurrence count is low, with at most {max_occ} occurrences per {win} points",
                "the pattern is too sporadic to keep under the current rule",
                "the behavior is treated as occasional with no more than {max_occ} occurrences in {win} points",
                "the low-frequency motif condition is met with at most {max_occ} occurrences in {win} points",
                "the repetition rate is too low for retention",
                "the pattern is judged as rare under the {win}-point window criterion",
                "the sparse occurrence pattern crosses the occasional-pattern filter condition"
            ],

            "ignore_short_duration": [
                "the duration {dur} is shorter than the required minimum {min_d} points",
                "the anomaly is too brief (duration {dur} < {min_d} points)",
                "the event lasts only {dur} points, below the minimum duration {min_d}",
                "the duration requirement of at least {min_d} points is not met (got {dur})",
                "the anomaly does not meet the minimum duration threshold {min_d} points",
                "the anomaly duration is under the configured minimum of {min_d} points (observed {dur})",
                "the case is transient and fails the minimum-duration constraint ({dur} < {min_d})",
                "it does not persist long enough; required >= {min_d} points but observed {dur}",
                "the event is shorter than the allowed minimum duration window ({min_d} points)",
                "the minimum duration gate rejects it: {dur} points is below {min_d}",
            ],
            "truncate_overlong_duration": [
                "the duration exceeds the allowed limit of {max_d} points",
                "the anomaly duration is over the limit of {max_d} points",
                "the length runs beyond the maximum of {max_d} points",
                "the span is longer than the cap of {max_d} points",
                "the current duration of {dur} points is greater than the allowed {max_d} points",
                "the observed duration surpasses the configured maximum of {max_d} points",
                "the anomaly exceeds the maximum permitted length of {max_d} points",
                "the rule is violated because the duration goes over {max_d} points",
                "the anomaly persists beyond the limit of {max_d} points",
                "the duration constraint is not satisfied since it exceeds {max_d} points"
            ],
            "ignore_recent_reoccurrence": [
                "a similar anomaly occurred within the last {win} points and its {ignore_recent_mode_or_nl} did not exceed historical ones",
                "it is a recent reoccurrence in a {win}-point window, and {ignore_recent_mode_or_nl} is not greater than past similar anomalies",
                "the anomaly repeats within {win} points but shows no stronger {ignore_recent_mode_or_nl} than historical counterparts",
                "a same-pattern event exists in the recent {win}-point history, and this one does not surpass it in {ignore_recent_mode_or_nl}",
                "this near-term repeat (within {win} points) is ignored because {ignore_recent_mode_or_nl} does not beat previous similar events",
                "it falls under recent reoccurrence filtering: within {win} points, {ignore_recent_mode_or_nl} is not above historical anomalies",
                "a recent similar anomaly is already present in the {win}-point window, and current {ignore_recent_mode_or_nl} does not exceed it",
                "the anomaly is a short-window repeat ({win} points) with non-superior {ignore_recent_mode_or_nl}, so it is filtered",
                "within the recent {win}-point range, this repeat does not show higher {ignore_recent_mode_or_nl} than previous same-pattern anomalies",
                "the recurrence is too close in time ({win} points) and lacks improvement in {ignore_recent_mode_or_nl} versus historical records"
            ]
        }

        # Diversified single-line phrasing for intent application per anomaly
        # (kept / excluded / not kept but no explicit rule attribution)
        self.intent_action_line_pool = {
            "kept": [
                " Kept as a final anomaly under the intent preference.",
                " Kept as a final anomaly after applying the user intent.",
                " Kept in the final results per the user's intent.",
                " Kept (matches the intent preference).",
                " Kept as final; it satisfies the intent constraints.",
                " Kept in the final set based on the intent.",
                " Kept as final according to the intent preference.",
                " Kept; it aligns with what the user cares about.",
                " Kept as final per the stated intent.",
                " Kept as a final anomaly after intent-based selection.",
            ],
            "excluded": [
                " Excluded by intent rule: {rule}.",
                " Excluded under the intent constraint: {rule}.",
                " Filtered out by intent ({rule}).",
                " Removed by the intent preference: {rule}.",
                " Dropped due to the intent constraint ({rule}).",
                " Not kept because of the intent rule: {rule}.",
                " Excluded after applying intent: {rule}.",
                " Pruned by intent: {rule}.",
                " Eliminated under the intent preference ({rule}).",
                " Excluded per the intent rule ({rule}).",
            ],
            "not_kept": [
                " Not kept in final results under the intent preference.",
                " Not kept after applying the intent preference.",
                " Not retained in the final results once intent is applied.",
                " Not kept as a final anomaly under the user's intent.",
                " Not retained after intent-based selection.",
                " Not kept in the final set under the intent constraints.",
                " Not kept in final results under the stated intent.",
                " Not included in the final results after intent filtering.",
                " Not kept after applying the user's preference.",
                " Not retained as final under the intent preference.",
            ],
        }

        # Compact DAP profile templates (seq_len + magnitude + baseline)
        self.ts_profile_triplet_pool = [
            "Sequence length is {seq_len}, magnitude is about {mag}, and baseline is around {baseline}.",
            "Series snapshot: {seq_len} points, magnitude near {mag}, baseline near {baseline}.",
            "This series has {seq_len} timesteps with fluctuation magnitude approximately {mag} and baseline approximately {baseline}.",
            "Length-magnitude-baseline summary: length={seq_len}, magnitude≈{mag}, baseline≈{baseline}.",
            "Global profile shows {seq_len} points, magnitude around {mag}, and a baseline around {baseline}.",
            "Quick profile: length={seq_len}, mag≈{mag}, baseline≈{baseline}.",
            "At a glance, the sequence spans {seq_len} points with magnitude about {mag} and baseline about {baseline}.",
            "Scale cue: length {seq_len}, magnitude near {mag}, baseline near {baseline}.",
            "Profile metrics indicate {seq_len} observations, an magnitude of roughly {mag}, and baseline level {baseline}.",
            "Series-level context: {seq_len} data points, magnitude≈{mag}, baseline≈{baseline}.",
        ]

    def generate_intent_result_line(self, rule: Dict[str, Any], anomaly: Dict[str, Any], duration: int) -> str:
        """Generate a diversified single-line message for an excluded/truncated anomaly.
        The line will include an action label (Excluded/Truncated) and a diversified reason.
        Rule description text is intentionally omitted to keep output concise.
        """
        rule_name = str(rule.get("name") or "")
        action_label = "Truncated" if rule_name == "truncate_overlong_duration" else "Excluded"
        action_lower = action_label.lower()
        action_verb = "truncate" if rule_name == "truncate_overlong_duration" else "exclude"
        action_noun = "truncation" if rule_name == "truncate_overlong_duration" else "exclusion"
        params = rule.get("parameters", {}) or {}
        pat = (anomaly.get("pattern_type") or "").lower()
        mag = anomaly.get("magnitude")
        # Format parameters for templates
        thr_str = params.get("threshold_value_str", str(params.get("threshold_value", "")))
        min_occ = int(params.get("min_occurrences", 3) or 3)
        max_occ = int(params.get("max_occurrences", 1) or 1)
        win = int(params.get("window_size", 100) or 100)
        ratio = float(params.get("relative_ratio_threshold", 1.0) or 1.0)
        ratio_d = float(params.get("relative_duration_ratio_threshold", 1.0) or 1.0)
        mode = str(params.get("mode", "both") or "both").strip().lower()
        ignore_recent_mode_and_nl = mode if mode != "both" else "magnitude and duration"
        ignore_recent_mode_or_nl = mode if mode != "both" else "magnitude or duration"
        max_d = int(params.get("max_duration", 50) or 50)
        st = int(anomaly.get("start_pos", 0) or 0)
        new_end = st + max(1, max_d) - 1
        dur = int(anomaly.get("end_pos", st) - st + 1)
        try:
            mag_str = f"{float(mag):.3g}" if mag is not None else "N/A"
        except Exception:
            mag_str = str(mag)

        # Pick reason template
        reason_tpls = self.intent_reason_pool.get(rule_name, ["does not satisfy the applied intent constraints"])
        reason_tpl = random.choice(reason_tpls)
        min_d = int(params.get("min_duration", 2) or 2)
        reason = reason_tpl.format(
            pat=pat,
            mag=mag_str,
            thr=thr_str,
            min_occ=min_occ,
            max_occ=max_occ,
            win=win,
            ratio=ratio,
            ratio_d=ratio_d,
            ignore_recent_mode_and_nl=ignore_recent_mode_and_nl,
            ignore_recent_mode_or_nl=ignore_recent_mode_or_nl,
            st=st,
            new_end=new_end,
            max_d=max_d,
            dur=dur,
            min_d=min_d,
        )
        # Diversified line templates for action + reason
        line_templates = [
            " This anomaly is {action_lower} because {reason}.",
            " We {action_verb} this anomaly given that {reason}.",
            " The anomaly is {action_lower} due to the reason that {reason}.",
            " We make the decision to {action_verb} it because {reason}.",
            " The outcome is {action_lower} because {reason}.",
            " We apply {action_noun} given that {reason}.",
            " We decide to {action_verb} it given that {reason}.",
            " It is {action_lower} given that {reason}.",
            " The final call is {action_lower} because {reason}.",
            " We proceed with {action_noun} given that {reason}.",
        ]
        tpl = random.choice(line_templates)
        base_line = tpl.format(action=action_label, action_lower=action_lower, action_verb=action_verb, action_noun=action_noun, reason=reason)

        # If this is a truncation action, append a concise result description of the kept window
        if rule_name == "truncate_overlong_duration":
            new_len = max(1, min(dur, max_d))
            trunc_result_templates = [
                " After truncation, we keep the window from {st} to {new_end}.",
                " We retain only the segment from {st} to {new_end} and the rest is not kept.",
                " The kept range runs from {st} to {new_end} and the remaining portion stays outside the result.",
                " We preserve the front section from {st} to {new_end} and leave the remainder out.",
                " The result keeps indices from {st} to {new_end} while the tail is not included.",
                " Only the first {new_len} points are kept from {st} to {new_end}.",
                " We keep the part spanning {st} to {new_end} and do not include the rest.",
                " The retained window covers {new_len} points from {st} to {new_end}.",
                " The output keeps the range from {st} to {new_end} and ignores the remainder.",
                " We keep the beginning portion ending at {new_end} starting from {st}.",
            ]
            trunc_tpl = random.choice(trunc_result_templates)
            return base_line + trunc_tpl.format(st=st, new_end=new_end, max_d=max_d, dur=dur, new_len=new_len)

        return base_line
        

    def generate_kept_line(self, updates: List[str]) -> str:
        """Generate a diversified natural-language line for kept anomalies.
        If baseline updates are present, weave them in with varied phrasing.
        """
        if updates:
            with_updates_templates = [
                " It is kept. We update the baseline accordingly. {updates}",
                " This anomaly is kept because it satisfies the intent constraints. We then update the baseline. {updates}",
                " We keep this anomaly and update the baseline accordingly. {updates}",
                " It is kept, followed by a baseline update. {updates}",
                " The anomaly is retained and the baseline is refreshed. {updates}",
                " We decide to keep it and make baseline changes. {updates}",
                " Criteria are satisfied so it is kept, and we update the baseline. {updates}",
                " It is retained and we refresh the baseline. {updates}",
                " It is preserved and we revise the baseline. {updates}",
                " Kept for future reference and we adjust the baseline. {updates}",
            ]
            joined = " ".join(updates)
            tpl = random.choice(with_updates_templates)
            return tpl.format(updates=joined)
        else:
            no_updates_templates = [
                " It is kept.",
                " This anomaly is kept because it satisfies the intent constraints.",
                " We keep this anomaly.",
                " It is kept as it satisfies the criteria.",
                " The anomaly is retained.",
                " We decide to keep it.",
                " It is retained for subsequent analysis.",
                " It is preserved for downstream steps.",
                " It remains in the result set.",
                " We keep it for further reference.",
            ]
            return random.choice(no_updates_templates)

    def generate_detection_summary(self, total_detected: int, final_remaining: Optional[int] = None, rule_count: int = 0) -> str:
        """Generate a diversified summary line for detection stage.

        If intent constraints exist (rule_count > 0), include the final remaining anomaly count.
        """
        det = int(total_detected or 0)
        has_intent = bool((rule_count or 0) > 0)
        rem = int(final_remaining or 0) if has_intent else None
        anom_det = "anomaly" if (det == 1 or det == 0) else "anomalies"
        be_verb = "is" if (det == 1 or det == 0) else "are"
        anom_rem = "anomaly" if (rem == 1 or rem == 0) else "anomalies"
        remain_verb = "remains" if (rem == 1 or rem == 0) else "remain"

        if has_intent:
            templates = [
                "We detect {det} {anom_det}; after applying the user intent, {rem} {remain_verb}.",
                "We detect {det} {anom_det} and apply intent constraints, leaving {rem} {anom_rem}.",
                "Detection finds {det} {anom_det}. After intent constraints, {rem} {anom_rem} {remain_verb}.",
                "There {be_verb} {det} {anom_det}. After intent constraints, {rem} {anom_rem} {remain_verb}.",
                "Total {det} {anom_det} {be_verb} detected; intent keeps {rem} {anom_rem}.",
                "We have {det} {anom_det}; intent constraints retain {rem} {anom_rem}.",
                "Initial detection shows {det} {anom_det}; after applying intent, {rem} {remain_verb}.",
                "We identify {det} {anom_det}. Intent constraints leave {rem} {anom_rem}.",
                "There {be_verb} {det} {anom_det}; after intent, {rem} {anom_rem} {remain_verb}.",
                "A total of {det} {anom_det} {be_verb} found; intent reduces to {rem} {anom_rem}.",
            ]
            tpl = random.choice(templates)
            return tpl.format(det=det, anom_det=anom_det, rem=rem, anom_rem=anom_rem, remain_verb=remain_verb, be_verb=be_verb)
        else:
            templates = [
                "We detect {det} {anom_det}.",
                "There {be_verb} {det} {anom_det}.",
                "Total {det} {anom_det} {be_verb} detected.",
                "Detection finds {det} {anom_det}.",
                "The anomaly count is {det}.",
                "A total of {det} {anom_det} {be_verb} detected.",
                "We find {det} {anom_det}.",
                "We identify {det} {anom_det}.",
                "We observe {det} {anom_det}.",
                "{det} {anom_det} {be_verb} in scope.",
            ]
            tpl = random.choice(templates)
            return tpl.format(det=det, anom_det=anom_det, be_verb=be_verb)

    def generate_human_intent_structured(self, human_intent: Dict[str, Any], ts_info: Dict[str, Any]) -> str:
        """Generate a human-readable but structured intent text.

        This is used for the model-visible reasoning trace (e.g. inside <think>) and for
        building prompts that later get rewritten into natural language.

        Note: we intentionally keep a *single* stable formatting style here (natural-language
        bullet list) to reduce variance and make it easier for models/tools to copy verbatim.
        """
        rules = (human_intent or {}).get("rules", []) or []
        if not rules:
            return "No specific intent constraints."

        # Single stable format: natural-language bullet list using generate_intent_description
        lines: List[str] = []
        kept_any = False
        for i, r in enumerate(rules):
            if not isinstance(r, dict):
                continue
            try:
                desc = self.generate_intent_description(r, ts_info)
            except Exception:
                desc = str(r.get("name") or "unknown")
            desc = str(desc or "").strip()
            if not desc:
                continue
            lines.append(f"{i+1}. {desc}")
            kept_any = True

        return "\n".join(lines) if kept_any else "No specific intent constraints."

    def generate_human_intent_nl_fallback(self, human_intent: Dict[str, Any]) -> str:
        """Fallback natural-language user intent (diversified)."""
        rules = (human_intent or {}).get("rules", []) or []
        if not rules:
            templates = [
                "No special preferences; please report all detected anomalies.",
                "I don't have extra constraints—just detect anomalies as usual.",
                "Please keep all detected anomalies in the final results.",
                "There are no specific intent rules; report all anomalies found.",
                "I have no additional intent—just provide the detected anomalies.",
                "Please include all detected anomalies without extra filtering.",
                "No particular intent—just return all anomalies you find.",
                "I don't have special rules; report every detected anomaly.",
                "Please keep all anomalies detected without extra constraints.",
                "There are no extra preferences; just show all detected anomalies.",
                "I have no special intent—report all anomalies as is.",
                "Please provide all detected anomalies without additional filtering.",
                "No specific intent—just return all anomalies found.",
                "I don't have extra rules; report all detected anomalies.",
                "Please include every detected anomaly without further constraints.",
            ]
            return random.choice(templates)
        # Mention rule names concisely
        names: List[str] = []
        for r in rules:
            if not isinstance(r, dict):
                continue
            n = r.get("name")
            if isinstance(n, str) and n:
                names.append(n)
        if not names:
            templates = [
                "Please follow the provided intent rules when deciding final anomalies.",
                "Apply the intent constraints to decide which anomalies matter.",
            ]
            return random.choice(templates)
        templates = [
            "Please decide the final anomalies according to these intent rules: {rules}.",
            "I only care about anomalies that match this intent: {rules}.",
            "Use the following intent constraints to refine the final results: {rules}.",
        ]
        return random.choice(templates).format(rules=", ".join(names))

    
    def generate_common_description(self, key, **kwargs):
        if key in self.common_desc_pool:
            desc_options = self.common_desc_pool[key]
            description = random.choice(desc_options)
            return description.format(**kwargs)
        else:
            return f"Unknown common description key: {key}"


    def generate_history_anomalies_description(
        self,
        idx_i: int,
        history_list: List[Any],
        pattern: str,
        start: int,
        window_size: int,
    ) -> str:
        """Render a deterministic tool response string for query_memory results."""

        def _fmt_mag(v: Any) -> str:
            try:
                return f"{float(v):.3g}"
            except Exception:
                return str(v)

        p = str(pattern or "unknown")
        try:
            st_i = int(start)
        except Exception:
            st_i = 0
        try:
            win_i = int(window_size)
        except Exception:
            win_i = 0

        if not isinstance(history_list, list) or len(history_list) == 0:
            return (
                f"Historical anomalies for ID: {idx_i} ({p}) | Start: {st_i} | Window: {win_i} points | Results: []"
            )

        valid_rows: List[str] = []
        for item in history_list:
            if not isinstance(item, (list, tuple)) or len(item) != 4:
                continue
            s, e, hp, m = item
            try:
                s_i = int(s)
                e_i = int(e)
            except Exception:
                continue
            if e_i < s_i:
                s_i, e_i = e_i, s_i
            valid_rows.append(
                f"({s_i}, {e_i}, {str(hp)}, {_fmt_mag(m)})"
            )

        if not valid_rows:
            return (
                f"Historical anomalies for ID: {idx_i} ({p}) | Start: {st_i} | Window: {win_i} points | Results: []"
            )

        return (
            f"Historical anomalies for ID: {idx_i} ({p}) | Start: {st_i} | Window: {win_i} points | Results: [" + ", ".join(valid_rows) + "]"
        )


    def generate_irregular_pattern_description(self, length, start_positions):
        positions_str = ", ".join(str(pos) for pos in start_positions)
        descriptions = [
            f"The irregular pattern has a length of {length} and appears at positions {positions_str}.",
            f"An irregular pattern is detected with a length of {length} at the following positions: {positions_str}.",
            f"Detected an irregular pattern of length {length} at positions: {positions_str}.",
            f"An irregular pattern spans {length} data points, occurring at positions: {positions_str}."
        ]
        return random.choice(descriptions)


    def generate_ts_characteristic_description(self, characteristic_dict, additional_info=None):
        description_dict = {
            "seasonality": "",
            "trend": "",
            "noise": ""
        }

        def _num(x):
            try:
                return float(x)
            except Exception:
                return None

        def _fmt(x):
            if x is None:
                return "unknown"
            try:
                return f"{float(x):.3g}"
            except Exception:
                return str(x)

        # Handle seasonality combination
        if 'seasonal' in characteristic_dict and 'frequency' in characteristic_dict:
            seasonal_pattern = characteristic_dict['seasonal']
            frequency_pattern = characteristic_dict['frequency']
            seasonality_key = (seasonal_pattern, frequency_pattern)

            if seasonality_key in self.ts_characteristic_desc_pool['seasonality']:
                desc_options = self.ts_characteristic_desc_pool['seasonality'][seasonality_key]
                description = random.choice(desc_options)
                description_dict["seasonality"] = description

            ts_additional = (additional_info or {}).get("additional_info", {}) if isinstance(additional_info, dict) else {}
            if characteristic_dict["seasonal"] == "irregular pattern":
                pattern_length = ts_additional.get("irregular_pattern_length")
                start_positions = ts_additional.get("irregular_pattern_start_list", [])
                if pattern_length is not None and isinstance(start_positions, list) and len(start_positions) > 0:
                    addition_str = self.generate_irregular_pattern_description(pattern_length, start_positions)
                    description_dict["seasonality"] += " " + addition_str

        # Handle other characteristics normally
        for characteristic, value in characteristic_dict.items():
            if characteristic in ['seasonal', 'frequency']:
                continue

            if characteristic in self.ts_characteristic_desc_pool and value in self.ts_characteristic_desc_pool[characteristic]:
                desc_options = self.ts_characteristic_desc_pool[characteristic][value]
                description = random.choice(desc_options)
                description_dict[characteristic] = description

        # Compact DAP summary (focus only on seq_len / magnitude / baseline)
        profile_lines: List[str] = []
        if isinstance(additional_info, dict):
            seq_len = additional_info.get("seq_len")
            magnitude = _num(additional_info.get("magnitude"))
            baseline = _num(additional_info.get("baseline_value"))

            if seq_len is not None or magnitude is not None or baseline is not None:
                tpl = random.choice(self.ts_profile_triplet_pool)
                profile_lines.append(
                    tpl.format(
                        seq_len=seq_len if seq_len is not None else "unknown",
                        mag=_fmt(magnitude),
                        baseline=_fmt(baseline),
                    )
                )

        # Put key numeric profiling BEFORE season/trend/noise descriptions
        final_descriptions = profile_lines + [description_dict["seasonality"], description_dict["trend"], description_dict["noise"]]
        return " ".join([d for d in final_descriptions if d])


    def generate_anomaly_description(self, anomaly_pattern):
        pattern_type = anomaly_pattern.get("pattern_type", "")
        start_pos = anomaly_pattern.get("start_pos", 0)
        end_pos = anomaly_pattern.get("end_pos", 0)
        duration = end_pos - start_pos + 1
        magnitude = anomaly_pattern.get("magnitude", 0.0)
        
        if pattern_type in self.anomaly_pattern_desc_pool:
            desc_options = self.anomaly_pattern_desc_pool[pattern_type]
            description = random.choice(desc_options)
            return description.format(
                duration=duration,
                start_pos=start_pos,
                end_pos=end_pos,
                magnitude=magnitude
            )
        else:
            # Fallback description for unknown pattern types
            return f"Unknown anomaly pattern: {pattern_type} with magnitude {magnitude:.3g} over {duration} points from position {start_pos} to {end_pos}"

    def _sampling_interval_seconds(self, ts_info: Optional[Dict[str, Any]]) -> int:
        if not isinstance(ts_info, dict):
            return 60
        raw = ts_info.get("sampling_interval", 60)
        try:
            val = int(raw)
        except Exception:
            val = 60
        return max(1, val)

    def _format_duration_compact(self, total_seconds: int) -> str:
        sec = max(0, int(total_seconds))
        hours, remainder = divmod(sec, 3600)
        minutes, seconds = divmod(remainder, 60)
        parts: List[str] = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if seconds > 0 or not parts:
            parts.append(f"{seconds}s")
        return " ".join(parts)

    def _build_window_parse_trace(self, parameters: Dict[str, Any], ts_info: Optional[Dict[str, Any]]) -> str:
        if not isinstance(parameters, dict) or "window_size" not in parameters:
            return ""
        try:
            win_points = int(parameters.get("window_size", 0) or 0)
        except Exception:
            return ""
        if win_points <= 0:
            return ""

        interval_seconds = self._sampling_interval_seconds(ts_info)
        total_seconds = win_points * interval_seconds
        duration_text = self._format_duration_compact(total_seconds)
        parsed_points = total_seconds // interval_seconds
        return (
            "Window parse: "
            f"sampling_interval={interval_seconds}s/point, "
            f"lookback_time={duration_text} ({total_seconds}s), "
            f"parsed_points={total_seconds}//{interval_seconds}={parsed_points}."
        )
    

    def generate_intent_description(self, rule, ts_info = None, include_window = True) -> str:
        rule_name = rule.get("name", "")
        evidence_samples = rule.get("evidence_samples", []) or []
        parameters = rule.get("parameters", {}) or {}
        rank_by_raw = str(parameters.get("rank_by", "magnitude") or "magnitude").strip().lower()
        mode = str(parameters.get("mode", "both") or "both").strip().lower()
        rank_by_nl_map = {
            "magnitude": "stronger anomaly magnitude",
            "duration": "longer-lasting anomaly duration",
            "hybrid": "a combined severity score that considers both anomaly magnitude and duration",
        }
        rank_by_nl = rank_by_nl_map.get(rank_by_raw, "overall anomaly importance")
        ignore_recent_mode_and_nl = mode if mode != "both" else "magnitude and duration"
        ignore_recent_mode_or_nl = mode if mode != "both" else "magnitude or duration"

        #                     description 
        desc_candidates = self.intent_rule_desc_pool.get(
            rule_name, self.intent_rule_desc_pool.get("_generic", ["Apply this intent criterion."])
        )

        fmt_params = {
            "threshold_value_str": parameters.get("threshold_value_str", str(parameters.get("threshold_value", ""))),
            "min_occurrences": int(parameters.get("min_occurrences", 3) or 3),
            "window_size": int(parameters.get("window_size", 100) or 100),
            "max_occurrences": int(parameters.get("max_occurrences", 1) or 1),
            "relative_ratio_threshold": float(parameters.get("relative_ratio_threshold", 1.0) or 1.0),
            "relative_duration_ratio_threshold": float(parameters.get("relative_duration_ratio_threshold", 1.0) or 1.0),
            "max_duration": int(parameters.get("max_duration", 50) or 50),
            "min_duration": int(parameters.get("min_duration", 2) or 2),
            "max_gap": int(parameters.get("max_gap", 1) or 1),
            "same_pattern_only": bool(parameters.get("same_pattern_only", True)),
            "count": int(parameters.get("count", 1) or 1),
            "min_duration_scope": int(parameters.get("min_duration", 20) or 20),
            "min_offset_ratio": float(parameters.get("min_offset_ratio", 0.0) or 0.0),
            "variance_ratio": float(parameters.get("variance_ratio", 1.0) or 1.0),
            "max_reports": int(parameters.get("max_reports", 3) or 3),
            "rank_by": rank_by_raw,
            "rank_by_nl": rank_by_nl,
            "ignore_recent_mode_and_nl": ignore_recent_mode_and_nl,
            "ignore_recent_mode_or_nl": ignore_recent_mode_or_nl,
        }

        try:
            desc = random.choice(desc_candidates).format(**fmt_params)
        except Exception:
            desc = random.choice(self.intent_rule_desc_pool.get("_generic", ["Apply this intent criterion."]))

        parts: List[str] = []
        window_parse = self._build_window_parse_trace(parameters=parameters, ts_info=ts_info)
        if include_window and window_parse:
            parts.append(window_parse)
        parts.append(f"{desc}")

        # def fmt_mag(v):
        #     try:
        #         return f"{float(v):.3g}"
        #     except Exception:
        #         return str(v)

        # if rule_name in ("ignore_recent_reoccurrence_with_small_magnitude", "ignore_recent_reoccurrence_with_short_duration"):
        #     if evidence_samples:
        #         parts.append("Historical evidence of recent reoccurrence:")
        #         for idx, ev in enumerate(evidence_samples, 1):
        #             pat = ev.get("pattern", "unknown")
        #             mag = ev.get("magnitude", None)
        #             dur = ev.get("duration", None)
        #             fields = [f"pattern {pat}"]
        #             if mag is not None:
        #                 fields.append(f"previous magnitude {fmt_mag(mag)}")
        #             if dur is not None:
        #                 fields.append(f"previous duration {int(dur)} points")
        #             parts.append(f"- {idx}. " + ", ".join(fields))

        # elif rule_name == "ignore_frequent_patterns":
        #     if evidence_samples:
        #         parts.append("Evidence of frequent/periodic patterns:")
        #         for idx, ev in enumerate(evidence_samples, 1):
        #             pat = ev.get("pattern", "unknown")
        #             mag = ev.get("magnitude", None)
        #             dur = ev.get("duration", None)
        #             evid = [f"pattern {pat}"]
        #             ev_id = ev.get("id")
        #             if ev_id:
        #                 evid.append(f"id {ev_id}")
        #             if mag is not None:
        #                 evid.append(f"magnitude {fmt_mag(mag)}")
        #             if dur is not None:
        #                 evid.append(f"duration {int(dur)} points")
        #             parts.append(f"- {idx}. " + ", ".join(evid))

        return " ".join(parts)

    def generate_intent_rule_brief(self, rule: Dict[str, Any]) -> str:
        """Generate a short, human-readable rule phrase.

        Used in inline per-anomaly attribution lines where we want a concise rule mention.
        """
        if not isinstance(rule, dict):
            return "intent rule"
        try:
            desc = self.generate_intent_description(rule, include_window=False)
        except Exception:
            desc = str(rule.get("name") or "intent rule")

        text = str(desc or "").strip()
        if not text:
            return str(rule.get("name") or "intent rule")

        # Keep it short: take the first sentence-like chunk.
        text = re.split(r"[\n\r]+", text)[0]
        text = re.split(r"(?<=[.!?])\s+", text)[0]
        text = text.strip(" \t\n\r.,:;-")

        # Light de-verbosifying for common leading verbs.
        text = re.sub(
            r"^(enforce|require|exclude|ignore|retain|keep|only keep|only consider|restrict|apply)\b\s*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip(" \t\n\r.,:;-")

        # # Hard cap length to keep inline attribution compact.
        # max_len = 100
        # if len(text) > max_len:
        #     text = text[: max_len - 3].rstrip() + "..."
        # Hard cap length by word count to keep inline attribution compact.
        max_words = 20
        words = text.split()
        if len(words) > max_words:
            text = " ".join(words[: max_words])
        return text or str(rule.get("name") or "intent rule")

    def _format_support_anomaly_compact(self, anomaly: Dict[str, Any]) -> str:
        try:
            s = int(anomaly.get("start_pos", 0) or 0)
            e = int(anomaly.get("end_pos", s) or s)
        except Exception:
            s = anomaly.get("start_pos", 0)
            e = anomaly.get("end_pos", s)
        p = str(anomaly.get("pattern_type") or "unknown")
        src = str(anomaly.get("source") or "")
        try:
            m = f"{float(anomaly.get('magnitude', 0.0)):.3g}"
        except Exception:
            m = str(anomaly.get("magnitude"))
        src_suffix = f", src={src}" if src else ""
        return f"({s},{e},{p},mag={m}{src_suffix})"

    def _format_support_anomaly_simple(self, anomaly: Dict[str, Any]) -> str:
        try:
            s = int(anomaly.get("start_pos", 0) or 0)
            e = int(anomaly.get("end_pos", s) or s)
        except Exception:
            s = anomaly.get("start_pos", 0)
            e = anomaly.get("end_pos", s)
        p = str(anomaly.get("pattern_type") or "unknown")
        try:
            m = f"{float(anomaly.get('magnitude', 0.0)):.3g}"
        except Exception:
            m = str(anomaly.get("magnitude"))
        return f"({s}, {e}, {p}, {m})"

    def _generate_complex_analysis_prefix(self, intent_action_context: Optional[Dict[str, Any]] = None) -> str:
        if not isinstance(intent_action_context, dict):
            return ""
        analysis = intent_action_context.get("analysis")
        if not isinstance(analysis, dict):
            return ""

        kind = str(analysis.get("analysis_kind") or "").strip().lower()
        support = analysis.get("support_anomalies") if isinstance(analysis.get("support_anomalies"), list) else []
        support_preview = ", ".join(self._format_support_anomaly_compact(a) for a in support[:3] if isinstance(a, dict))
        if len(support) > 3:
            support_preview += f", ... total={len(support)}"

        if kind == "ignore_frequency":
            win = int(analysis.get("window_size", 0) or 0)
            max_count = int(analysis.get("max_count", 0) or 0)
            comparison = str(analysis.get("comparison") or "")
            best_window = analysis.get("best_window") if isinstance(analysis.get("best_window"), dict) else {}
            left = best_window.get("start")
            right = best_window.get("end")
            criterion_text = comparison or "the configured frequency threshold"
            templates = [
                " After checking windows of {win} points that include this anomaly, the densest window is from {left} to {right}, where we see {max_count} same-pattern anomalies; this is compared against {criterion_text}",
                " Looking around this anomaly with a {win}-point horizon, the highest concentration appears in [{left}, {right}] with {max_count} matching data anomalies, then we judge it against {criterion_text}",
                " We scan all {win}-point windows covering the current point and find the strongest repetition in [{left}, {right}] with {max_count} occurrences, which we evaluate against {criterion_text}",
                " In the local frequency check, the window [{left}, {right}] contains the most same-pattern anomalies ({max_count}) among all {win}-point candidates that include this anomaly, and we compare that to {criterion_text}",
                " For this anomaly, the most crowded {win}-point segment is [{left}, {right}] with {max_count} matching data anomalies, and that count is measured against {criterion_text}",
                " Within windows of {win} points anchored around this anomaly, [{left}, {right}] gives the peak count ({max_count}); we then apply {criterion_text}",
                " After sliding a {win}-point window across positions that still include the current anomaly, the maximum repetition appears in [{left}, {right}] with {max_count} data anomalies, checked by {criterion_text}",
                " The nearest high-density area is [{left}, {right}] under a {win}-point view, containing {max_count} same-pattern anomalies, and we evaluate this using {criterion_text}",
                " We first locate the heaviest local cluster in a {win}-point neighborhood: [{left}, {right}] with {max_count} data anomalies, then decide based on {criterion_text}",
                " From the frequency perspective, [{left}, {right}] is the best supporting window in the {win}-point search, with {max_count} matching anomalies, and the decision follows {criterion_text}",
            ]
            msg = random.choice(templates).format(
                win=win,
                max_count=max_count,
                criterion_text=criterion_text,
                left=left,
                right=right,
            )
            return msg + "."

        if kind == "ignore_recent_reoccurrence":
            win = int(analysis.get("window_size", 0) or 0)
            mode = str(analysis.get("mode") or "both")
            matched_len = int(analysis.get("matched_prefix_len", 0) or 0)
            stop_reason = str(analysis.get("stop_reason") or "")
            decisive = analysis.get("decisive_reference")
            current = intent_action_context.get("current_anomaly") if isinstance(intent_action_context.get("current_anomaly"), dict) else {}
            decisive_str = ""
            if isinstance(decisive, dict):
                decisive_str = self._format_support_anomaly_simple(decisive)

            current_compare_str = ""
            if isinstance(current, dict):
                c_start = int(current.get("start_pos", 0) or 0)
                c_end = int(current.get("end_pos", c_start) or c_start)
                c_dur = c_end - c_start + 1
                try:
                    c_mag = f"{float(current.get('magnitude', 0.0)):.3g}"
                except Exception:
                    c_mag = str(current.get("magnitude"))

                if mode == "magnitude":
                    current_compare_str = f"the current magnitude is {c_mag}"
                elif mode == "duration":
                    current_compare_str = f"the current duration is {c_dur}"
                else:
                    current_compare_str = f"the current duration is {c_dur} and the current magnitude is {c_mag}"

            if matched_len <= 0:
                matched_phrase = "no comparable historical anomalies"
            elif matched_len == 1:
                matched_phrase = "1 comparable historical anomaly"
            else:
                matched_phrase = f"{matched_len} comparable historical anomalies"

            if mode == "magnitude":
                criterion_text = "whether the current magnitude is stronger than recent same-pattern events"
            elif mode == "duration":
                criterion_text = "whether the current duration lasts longer than recent same-pattern events"
            else:
                criterion_text = "whether both magnitude and duration are stronger than recent same-pattern events"

            stop_reason_text_map = {
                "first_non_matching_reference": [
                    "we reach the first historical anomaly that no longer supports filtering",
                    "the first conflicting historical anomaly appears",
                    "the chain breaks at the first non-supportive historical anomaly",
                    "we encounter a prior anomaly that fails the recurrence condition",
                ],
                "matched_all_in_window": [
                    "all comparable events inside the look-back window support the same conclusion",
                    "every eligible historical reference in the window points to the same decision",
                    "the full set of comparable recent events is consistent with this judgment",
                    "all nearby candidates within the window agree with the recurrence logic",
                    "none of the comparable windowed events contradicts the current conclusion",
                ],
                "no_previous_reference": [
                    "there is no comparable historical anomaly in the look-back window",
                    "no eligible historical reference exists in the recent window",
                    "the look-back range contains no same-pattern anomaly to compare against",
                    "we find no prior comparable anomaly within the configured history span",
                    "the recent history window is empty for this type of comparison",
                ],
            }
            stop_reason_text_candidates = stop_reason_text_map.get(stop_reason, ["the backward scan reaches its stopping condition"])
            stop_reason_text = random.choice(stop_reason_text_candidates)

            templates = [
                " We trace backward over the last {win} points and continuously compare {criterion_text}; {matched_phrase} appear before we stop because {stop_reason_text}",
                " Looking back {win} points from this anomaly, we compare each recent same-pattern anomaly by {criterion_text}; {matched_phrase} align with this judgment, and then we stop since {stop_reason_text}",
                " In the recent-history pass, we walk backward within {win} points and judge anomalies by {criterion_text}; the scan reaches {matched_phrase} before it stops because {stop_reason_text}",
                " To judge recent recurrence, we move from newer to older anomalies in a {win}-point window using {criterion_text}; {matched_phrase} support the trend, and we stop because {stop_reason_text}",
                " Backtracking over a {win}-point look-back, the recurrence logic checks {criterion_text}; we observe {matched_phrase} before stopping due to {stop_reason_text}",
                " We review the nearest same-pattern history within {win} points and compare by {criterion_text}; the process reaches {matched_phrase} before ending because {stop_reason_text}",
                " During recurrence reasoning, we scan backward in a {win}-point horizon and apply {criterion_text}; {matched_phrase} are found, then the scan stops as {stop_reason_text}",
                " The recent-repeat analysis inspects historical neighbors within {win} points using {criterion_text}; the scan yields {matched_phrase} before stopping because {stop_reason_text}",
                " We perform a backward recurrence check over {win} points and compare anomalies by {criterion_text}; this leads to {matched_phrase}, then we stop because {stop_reason_text}",
                " From the current anomaly, we look back {win} points and apply {criterion_text} to each candidate; we see {matched_phrase} before we stop as {stop_reason_text}",
            ]
            msg = random.choice(templates).format(
                win=win,
                criterion_text=criterion_text,
                matched_phrase=matched_phrase,
                stop_reason_text=stop_reason_text,
            )
            if stop_reason == "first_non_matching_reference" and decisive_str:
                if current_compare_str:
                    msg += f". The decisive historical anomaly is {decisive_str}; {current_compare_str}"
                else:
                    msg += f". The decisive historical anomaly is {decisive_str}"
            return msg + "."

        return ""

    def generate_intent_action_line(
        self,
        action: str,
        rule: Optional[Dict[str, Any]] = None,
        intent_action_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a diversified one-line suffix for per-anomaly intent application.

        For complex intent rules (ignore_frequency / ignore_recent_reoccurrence),
        prepend compact analysis evidence when intent_action_context is provided.
        """
        pool = self.intent_action_line_pool.get(action) or []
        if not pool:
            # Safe fallback (should rarely happen)
            if action == "excluded":
                brief = self.generate_intent_rule_brief(rule or {})
                base = f" Excluded by intent rule: {brief}."
                prefix = self._generate_complex_analysis_prefix(intent_action_context)
                return f"{prefix}{base}" if prefix else base
            if action == "not_kept":
                base = " Not kept in final results under the intent preference."
                prefix = self._generate_complex_analysis_prefix(intent_action_context)
                return f"{prefix}{base}" if prefix else base
            base = " Kept as a final anomaly under the intent preference."
            prefix = self._generate_complex_analysis_prefix(intent_action_context)
            return f"{prefix}{base}" if prefix else base

        tpl = random.choice(pool)
        if "{rule}" in tpl:
            brief = self.generate_intent_rule_brief(rule or {})
            base = tpl.format(rule=brief)
        else:
            base = tpl

        prefix = self._generate_complex_analysis_prefix(intent_action_context)
        return f"{prefix}{base}" if prefix else base
    

    def _generate_intent_reason(self, rule_name, rule_category, parameters, anomaly, duration):
        """Generate specific reason why an anomaly was excluded based on the latest intent rule set"""
        pattern_type = anomaly.get("pattern_type", "")
        anomaly_magnitude = anomaly.get("magnitude", 0)

        # Focus-based intent constraints (latest names)
        if rule_name == "focus_on_increase_only":
            decrease_patterns = ["sudden_decrease", "slow_decrease", "dip", "multi_dip", "zero_values"]
            if pattern_type in decrease_patterns:
                return "we only focus on increase-type anomalies, but the detected pattern is a decrease or non-increase type"
            return "the detected pattern does not match the increase-only focus"

        if rule_name == "focus_on_decrease_only":
            increase_patterns = ["sudden_increase", "slow_increase", "spike", "multi_spike"]
            if pattern_type in increase_patterns:
                return "we only focus on decrease-type anomalies, but the detected pattern is an increase type"
            return "the detected pattern does not match the decrease-only focus"

        if rule_name == "focus_on_zero_only":
            if pattern_type != "zero_values":
                return "we only focus on zero-value anomalies, but the detected pattern is not zero-values"

        # Threshold-based intent constraints
        if rule_name == "absolute_magnitude_threshold":
            threshold_value = parameters.get("threshold_value", 0)
            threshold_value_str = parameters.get("threshold_value_str", str(threshold_value))
            if anomaly_magnitude < threshold_value:
                return f"we only report anomalies with magnitude >= {threshold_value_str}, but this anomaly's magnitude is {anomaly_magnitude:.3g}"
            return f"the anomaly magnitude {anomaly_magnitude:.3g} does not satisfy the threshold of {threshold_value_str}"

        # Ignore frequency/periodic patterns
        if rule_name == "ignore_frequent_patterns":
            min_occ = parameters.get("min_occurrences", 3)
            window_size = parameters.get("window_size", 100)
            return f"this pattern occurs frequently/periodically (>= {min_occ} times within a {window_size}-point window), so it's ignored"

        if rule_name == "ignore_rare_patterns":
            max_occ = parameters.get("max_occurrences", 1)
            window_size = parameters.get("window_size", 100)
            return f"this pattern occurs only occasionally, with no more than {max_occ} occurrences within a {window_size}-point window, so it's ignored"

        # Ignore short duration
        if rule_name == "ignore_short_duration":
            min_d = int(parameters.get("min_duration", 2) or 2)
            return f"the anomaly duration is shorter than the minimum requirement ({duration} < {min_d} points)"

        # Recent reoccurrence – small magnitude
        if rule_name == "ignore_recent_reoccurrence_with_small_magnitude":
            ratio = parameters.get("relative_ratio_threshold", 1.0)
            return f"a similar anomaly occurred recently; current magnitude is not sufficiently larger than the previous (<= {ratio:.2f}x), thus ignored"

        # Recent reoccurrence – short duration
        if rule_name == "ignore_recent_reoccurrence_with_short_duration":
            ratio_d = parameters.get("relative_duration_ratio_threshold", 1.0)
            return f"a similar anomaly occurred recently; current duration is not sufficiently longer than the previous (<= {ratio_d:.2f}x), thus ignored"

        # Truncate overly long duration
        if rule_name == "truncate_overlong_duration":
            max_d = parameters.get("max_duration", 50)
            return f"the anomaly lasts over {max_d} points; only the first {max_d} points are kept"

        # Default fallback for other/legacy rules
        return f"the anomaly does not meet the criteria defined by the '{rule_name}' rule"


    def generate_baseline_update_description(self, kind: str, pattern: str, old_value, new_value, ratio_threshold: Optional[float] = None) -> str:
        """Generate diversified text describing baseline updates for recent reoccurrence rules.

        kind: 'magnitude' or 'duration'
        pattern: anomaly pattern type
        old_value/new_value: previous and updated baseline values
        ratio_threshold: optional threshold used by the rule
        """
        def fmt_mag(v):
            try:
                return f"{float(v):.3g}"
            except Exception:
                return str(v)

        def fmt_dur(v):
            try:
                return f"{int(v)} points"
            except Exception:
                return f"{v} points"

        if kind == "magnitude":
            old_str = "N/A" if old_value is None else fmt_mag(old_value)
            new_str = fmt_mag(new_value)
            thr_note = (
                f" and it is more than {ratio_threshold:.2f} times the previous baseline"
                if ratio_threshold is not None else ""
            )
            templates = [
                "The baseline magnitude for {p} is updated from {o} to {n}{t}.",
                "We update the magnitude baseline for {p} from {o} to {n}{t}.",
                "Set the magnitude baseline for {p} to {n}, previously {o}{t}.",
                "The magnitude reference for {p} moves from {o} to {n}{t}.",
                "We change the {p} magnitude baseline from {o} to {n}{t}.",
                "The magnitude baseline on {p} is now {n} and it was {o} before{t}.",
                "We record {n} as the magnitude baseline for {p}, replacing {o}{t}.",
                "We adopt {n} as the new magnitude baseline for {p}. The previous value was {o}{t}.",
                "The magnitude baseline for {p} becomes {n} from {o}{t}.",
                "We refresh the magnitude baseline for {p} to {n}. It was {o} earlier{t}.",
            ]
            tpl = random.choice(templates)
            return tpl.format(p=pattern, o=old_str, n=new_str, t=thr_note)

        # duration
        old_str = "N/A" if old_value is None else fmt_dur(old_value)
        new_str = fmt_dur(new_value)
        thr_note = (
            f" and it is more than {ratio_threshold:.2f} times the previous baseline"
            if ratio_threshold is not None else ""
        )
        templates = [
            "The baseline duration for {p} is updated from {o} to {n}{t}.",
            "We update the duration baseline for {p} from {o} to {n}{t}.",
            "Set the duration baseline for {p} to {n}, previously {o}{t}.",
            "The duration reference for {p} moves from {o} to {n}{t}.",
            "We change the {p} duration baseline from {o} to {n}{t}.",
            "The duration baseline on {p} is now {n} and it was {o} before{t}.",
            "We record {n} as the duration baseline for {p}, replacing {o}{t}.",
            "We adopt {n} as the new duration baseline for {p}. The previous value was {o}{t}.",
            "The duration baseline for {p} becomes {n} from {o}{t}.",
            "We refresh the duration baseline for {p} to {n}. It was {o} earlier{t}.",
        ]
        tpl = random.choice(templates)
        return tpl.format(p=pattern, o=old_str, n=new_str, t=thr_note)