import './App.css';
import React, { useEffect, useState,useContext  } from 'react';
import { AppBar, Radio, RadioGroup, FormControlLabel,FormControl, FormLabel, TextField, Typography, Button, Toolbar } from "@mui/material";
// import { DataGrid } from "@mui/x-data-grid";

function App() {
  const [metric, setMetric] = React.useState('jaccard');
  const [hashtag, setHashtag] = React.useState('');
  const [date, setDate] = React.useState('');
  const [location, setLocation] = React.useState('');



  const handleChange = (event) => {
    setMetric(event.target.value);
  };
 

  return (
    <div className="App">
    <div className = "App-Body">
    <AppBar>
                <Toolbar>
                    <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
                        Tweelink
                    </Typography>
                    {/* {loading? <CircularProgress color="inherit" />: ""}
                    {pages.map((page) => (
                        <Button
                        key={page}
                        onClick={() => openPage(page)}
                        sx={{ color: 'white', display: 'block' }}
                        >
                        {page}
                        </Button>
                    ))} */}
                </Toolbar>
            </AppBar>
  <div className="Input">
      <div className="InputOp">
        <TextField className="Input-Field" variant="standard" label="Hashtag" onChange={(e) => setHashtag(e.target.value)} type = "text"/>
        <TextField className="Input-Field" variant="standard" label="Date (YYYY-MM-DD)" onChange={(e) => setDate(e.target.value)}  type = "text"/>
        <TextField className="Input-Field" variant="standard" label="Location" type = "text" onChange={(e) => setLocation(e.target.value)}/>
       </div>
      <div className="InputOp">
        <FormControl>
          <FormLabel id="demo-controlled-radio-buttons-group">Evaluation Metric</FormLabel>
          <RadioGroup
            aria-labelledby="demo-controlled-radio-buttons-group"
            name="controlled-radio-buttons-group"
            value={metric}
            onChange={handleChange}
          >
            <FormControlLabel value="jaccard" control={<Radio />} label="Jaccard Coefficient" />
            <FormControlLabel value="tf_idf" control={<Radio />} label="TF-IDF" />
            <FormControlLabel value="cosine" control={<Radio />} label="Cosine similarity" />
            <FormControlLabel value="soft_cosine" control={<Radio />} label="Soft-cosine similarity" />
            <FormControlLabel value="bip" control={<Radio />} label="Binary Independence Model" />
            <FormControlLabel value="best_metric" control={<Radio />} label="Best evaluation metric" />
          </RadioGroup>
        </FormControl>

        

        </div>
        <Button variant="outlined">Submit</Button>
        </div>
        <div>
          <p>https://news.yahoo.com/kanye-west-targets-pete-davidson-093231036.html</p>
          <p>https://hollywoodlife.com/2022/02/13/kanye-west-pete-davidson-diss-captain-america-poster-photo/</p>
          <p>https://headlineplanet.com/home/2022/02/14/ed-sheeran-taylor-swifts-the-joker-and-the-queen-ranks-as-hot-adult-contemporary-radios-most-added-song/</p>
          <p>https://list.co.uk/news</p>
          <p>https://www.lifestyleasia.com/kl/culture/entertainment/forbes-highest-paid-entertainers-2022/</p>
        </div>
</div>
     
    </div>
  );
}

export default App;
