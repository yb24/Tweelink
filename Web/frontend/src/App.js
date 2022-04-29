import './App.css';
import React, { useEffect, useState,useContext  } from 'react';
import { AppBar, Radio, RadioGroup, FormControlLabel,FormControl, FormLabel, TextField, Typography, Button, Toolbar, CircularProgress } from "@mui/material";
import { DataGrid } from "@mui/x-data-grid";
import env from "react-dotenv";
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { DesktopDatePicker } from '@mui/x-date-pickers/DesktopDatePicker';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';
import DialogTitle from '@mui/material/DialogTitle';



function App() {
  const [loading, setLoading] = useState(false)

  const [metric, setMetric] = React.useState('jaccard');
  const [hashtag, setHashtag] = React.useState('');
  const [date, setDate] = React.useState('2022-02-14');
  const [location, setLocation] = React.useState('');

  const [model, setModel] = React.useState('');
  const [keyword, setKeyword] = React.useState('');

  const [data, setData] = React.useState([]);

  const [rows, setRows] = React.useState([]);
  
  //Datepicker
  const [value, setValue] = React.useState(new Date('2022-02-14'));

  const [search, setSearch] = React.useState('');

  const handleChange = (newValue) => {
    setValue(newValue);
    setDate(newValue.toISOString().slice(0, 10))
    
  };


  //For dialog box
  const [open, setOpen] = React.useState(false);

  const handleClickOpen = () => {
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
  };



 
  // React.useEffect(()=>{
  //   console.log(model);
  // },[model])

  // React.useEffect(()=>{
  //   console.log(keyword);
  // },[keyword])

  const columns = [
    {field: 'Rank', headerName: 'Rank', width: 40},
    { field: 'Date', headerName: 'Date', width: 180 },
    {
      field: 'Location',
      headerName: 'Location',
      width: 400,
      editable: true,
    },
    {
      field: 'Article',
      headerName: 'Article Link',
      width: 2000,
      editable: true,
      renderCell: (params) => (
        <a href={params.value}>{params.value}</a>
      )
    }
  
  ];

  function getResults(){

    if(model=='best_metric'){
      setKeyword('KECNW');
    }
  
    if(hashtag==''||date==''||location==''||model=='' || keyword=='' || model=='tf_idf' ||model=='cosine'){
      return;
    }

    setLoading(true);
    fetch(env.BACKEND+ `/result?hashtag=${hashtag}&date=${date}&location=${location}&model=${model}&keyword=${keyword}`)
    .then(response => response.json())
    .then((data) => {
      setLoading(false);  
      if(data.length!=0 && (data[0]=='Hashtag not found, Google Search Instead?' ||  data[0]==['Date out of range, Google Search Instead?'])){
        
        console.log("Invalid Input");
        setData([]);
        setRows([]);
        setSearch(data[0]);
        handleClickOpen();
      }
      else{
        setData(data);
        let rows_temp = []
        for(let i=0; i<data.length; i++){
          rows_temp.push({id: i+1, Rank: i+1, Date: data[i][2], Location: data[i][3].toUpperCase(), Article: data[i][4]})
        }
        setRows(rows_temp);
      }
      
    }).catch((e) => {
      setLoading(false);          
      console.log(e)
    }).catch((e) => {
      setLoading(false);          
      console.log(e)
    })
  }

  
  


  const handleChanges = (event) => {
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
                    {loading? <CircularProgress color="inherit" />: ""}
                    
                </Toolbar>
            </AppBar>
  <div className = "Extra">

  
  <div className="Input">
      <div className="InputOp">
      
        <TextField className="Input-Field" variant="standard" label="Hashtag" onChange={(e) => setHashtag(e.target.value)} type = "text"/>
        
        <TextField className="Input-Field" variant="standard" label="Location" type = "text" onChange={(e) => setLocation(e.target.value)}/>

      <div className="date">
      <LocalizationProvider dateAdapter={AdapterDateFns}>

      <DesktopDatePicker
        label="Date desktop"
        inputFormat="MM/dd/yyyy"
        value={value}
        onChange={handleChange}
        renderInput={(params) => <TextField {...params} />}
      />
    </LocalizationProvider>
      </div>
       
       </div>
      <div className="InputOp">
        <FormControl>
          <FormLabel id="demo-controlled-radio-buttons-group">Evaluation Metric</FormLabel>
          <RadioGroup
            aria-labelledby="demo-controlled-radio-buttons-group"
            name="controlled-radio-buttons-group"
            value={model}
            onChange={handleChanges}
          >
            <FormControlLabel value="jaccard" control={<Radio />} label="Jaccard Coefficient" onChange={(e) => setModel(e.target.value)}/>
            <FormControlLabel value="tf_idf" control={<Radio />} label="TF-IDF" onChange={(e) => setModel(e.target.value)}/>
            {model=="tf_idf" || model=="tf_idf1" || model=="tf_idf2" || model=="tf_idf3" || model=="tf_idf4" || model=="tf_idf5"?


            (<div className="InputOp2">
              <FormControlLabel value="tf_idf1" control={<Radio />} label="Binary" onChange={(e) => setModel(e.target.value)}/>
            <FormControlLabel value="tf_idf2" control={<Radio />} label="Double Normalization" onChange={(e) => setModel(e.target.value)}/>
            <FormControlLabel value="tf_idf3" control={<Radio />} label="Log Normalization" onChange={(e) => setModel(e.target.value)}/>
            <FormControlLabel value="tf_idf4" control={<Radio />} label="Raw Count"onChange={(e) => setModel(e.target.value)} />
            <FormControlLabel value="tf_idf5" control={<Radio />} label="Term Frequency" onChange={(e) => setModel(e.target.value)}/>
            </div>)
            :""}

            <FormControlLabel value="cosine" control={<Radio />} label="Cosine similarity" onChange={(e) => setModel(e.target.value)}/>
            {model=="cosine" || model=="cosine_count" || model=="cosine_tf"?
              (<div className="InputOp2">
              <FormControlLabel value="cosine_count" control={<Radio />} label="Count Vectorizer" onChange={(e) => setModel(e.target.value)}/>
            <FormControlLabel value="cosine_tf" control={<Radio />} label="TF-IDF Vectorizer" onChange={(e) => setModel(e.target.value)}/>
            </div>):""}

            <FormControlLabel value="soft_cosine" control={<Radio />} label="Soft-cosine similarity" onChange={(e) => setModel(e.target.value)}/>
            <FormControlLabel value="bip" control={<Radio />} label="Binary Independence Model" onChange={(e) => setModel(e.target.value)}/>
            <FormControlLabel value="bm25" control={<Radio />} label="BM25" onChange={(e) => setModel(e.target.value)}/>
            <FormControlLabel value="best_metric" control={<Radio />} label="Best evaluation metric" onChange={(e) => setModel(e.target.value)}/>
          </RadioGroup>
        </FormControl>

        


        </div>

        <Button id= "button_submit" variant="outlined" onClick={getResults}>Submit</Button>
      </div>

    {model!='best_metric'?
    (


      <div className="InputOp3">
      <FormControl>
          <FormLabel id="demo-controlled-radio-buttons-group">Keyword Extractor</FormLabel>
          <RadioGroup
            aria-labelledby="demo-controlled-radio-buttons-group"
            name="controlled-radio-buttons-group"
            value={keyword}
            onChange={handleChanges}
          >
     <FormControlLabel value="none" control={<Radio />} label="None" onChange={(e) => setKeyword(e.target.value)}/>
 <FormControlLabel value="KECNW" control={<Radio />} label="KECNW" onChange={(e) => setKeyword(e.target.value)}/>
      <FormControlLabel value="YAKE" control={<Radio />} label="YAKE" onChange={(e) => setKeyword(e.target.value)}/>
      <FormControlLabel value="RAKE" control={<Radio />} label="RAKE" onChange={(e) => setKeyword(e.target.value)}/>
      <FormControlLabel value="TextRank" control={<Radio />} label="TextRank"onChange={(e) => setKeyword(e.target.value)} />
      <FormControlLabel value="KeyBert" control={<Radio />} label="KeyBert"onChange={(e) => setKeyword(e.target.value)} />
      </RadioGroup>
          </FormControl>
      </div>
) :""}
      <div className="ReportsTable">
                <DataGrid 
                rows={rows}
                columns={columns}
                pageSize={10}
                rowsPerPageOptions={[10]}

                 
                />
                
            </div>
      </div>
      <Dialog
        open={open}
        onClose={handleClose}
        aria-labelledby="alert-dialog-title"
        aria-describedby="alert-dialog-description"
      >
        <DialogTitle id="alert-dialog-title">
          {search}
        </DialogTitle>
        <DialogContent>
          <DialogContentText id="alert-dialog-description">
            <a href={`https://www.google.com/search?q=${hashtag}+${location}+${date}&tbm=nws`}> Google Search</a>
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleClose}>Cancel</Button>
         
        </DialogActions>
      </Dialog>

  </div>
  
    
    </div>
  );
}

export default App;
